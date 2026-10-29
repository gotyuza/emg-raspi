"""
個人モデルの比較：モデルの種類と「複雑さ（大きさ）」を段階的に変えて、精度と計算量を比べる。
esn_putemg.py・esn_calcurve.py と同じフォルダに置いて使う。必要: pip install torch

■ 問い
  ・特徴量（log RMS / log WL）＋軽いモデルで十分か、深層学習は必要か
  ・深層学習を大きく（ノード数・層数を増やす）すると精度は上がるか
  ・軽い ESN を超えられないなら、計算を増やす必要はない

■ データ（1 日目だけ）
  入力  : 筋電 24ch（--band で変更可）。特徴量は 150ms 窓・20ms ごと（全モデル共通）
  正解  : 力センサー 10ch（ゼロ点を引き、MVC で割った %MVC）。出力も 10ch
  学習  : repeats_long を 10 秒ごとのかたまりに分け、5 つに 4 つ
  検証  : 同じく 5 つに 1 つ（早期終了とハイパーパラメータ選び）  ※時間順の 80/20 だと指が偏るため
  テスト: repeats_short 全部
  --day2: 2 日目の repeats_short でもテスト（校正なし。筋電の標準化だけ 2 日目の repeats_long で）

■ モデルと大きさ（S / M / L / XL）
  ridge       : 特徴量 → 線形（大きさなし）                         … 一番単純な基準
  esn         : 特徴量 → リザーバー 100 / 300 / 1000 / 3000
  lstm        : 特徴量 → LSTM 隠れ 32×1層 / 64×2 / 128×2 / 256×3
  cnnlstm     : 生波形 150ms → CNN(ch 16/32/64/128) → LSTM(32/64/128/256)
  transformer : 特徴量の過去 1 秒 → Transformer 次元 32×1層 / 64×2 / 128×2 / 128×4

■ 公平にするための決まり
  ・前処理、窓、ずらし幅、データの分け方は全モデル同じ
  ・ハイパーパラメータ探しの回数（--trials）は全モデル同じ。選ぶのは検証データだけ
  ・一番良い設定を、乱数の種（--seeds）を変えて学習し直し、平均とばらつきを出す

■ 評価
  NRMSE = RMSE ÷ 学習データでのそのセンサーの幅（1〜99% 点の差。最低 5%MVC）。10ch の平均
  ほかに 相関 r、MAE[%MVC]、遅れ[ms]、パラメータ数、1 ステップの推論時間（このPC・1 スレッド）

速さ: NVIDIA の GPU があれば自動で使う（--device cpu で使わない）。推論時間はいつも CPU・1 スレッドで測る。
      前処理の結果は cache/ フォルダに保存し、2 回目からはそれを読む。ESN は疎行列で計算する。

使い方（putemg-downloader フォルダで）:
  python personal_compare.py --quick                              # 動作確認（小さいモデルだけ・短く）
  python personal_compare.py                                      # 全モデル・全サイズ
  python personal_compare.py --models esn lstm --sizes S M        # 一部だけ
  python personal_compare.py --subjects 03 04 05 --seeds 0 1 2 --trials 5 --day2

結果: personal_<band>[_<tag>].csv（全部の数値）、.png（大きさと精度のグラフ）
      分けて実行するときは --tag light / --tag heavy のように名前を付けると上書きされない
"""
import argparse, copy, os, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import sosfilt
import torch
import torch.nn as nn

import esn_putemg as E
from esn_calcurve import split_blocks

NOUT = 10
SEQ_T, BURN, CTX = 100, 10, 50          # 学習の切り出し 2 秒 / 最初の 10 ステップは誤差に入れない / Transformer は過去 1 秒
SIZES = ["S", "M", "L", "XL"]
SPEC = {
    "ridge": {"S": {}},
    "esn": {"S": dict(n=100), "M": dict(n=300), "L": dict(n=1000), "XL": dict(n=3000)},
    "lstm": {"S": dict(h=32, l=1), "M": dict(h=64, l=2), "L": dict(h=128, l=2), "XL": dict(h=256, l=3)},
    "cnnlstm": {"S": dict(c=16, h=32), "M": dict(c=32, h=64), "L": dict(c=64, h=128), "XL": dict(c=128, h=256)},
    "transformer": {"S": dict(d=32, l=1), "M": dict(d=64, l=2), "L": dict(d=128, l=2), "XL": dict(d=128, l=4)},
}
# ハイパーパラメータの候補（試す回数は全モデル同じ）
HP_DEEP = dict(lr=[3e-4, 1e-3, 3e-3], drop=[0.0, 0.1, 0.3], wd=[1e-5, 1e-4, 1e-3])
HP_ESN = dict(rho=[0.5, 0.8, 0.95], in_scale=[0.1, 0.3, 1.0], leak=[0.3, 0.6, 1.0])
LAMS = [1e-2, 1e-1, 1, 10, 100, 1000]
DEV = torch.device("cpu")              # main() で GPU があれば cuda に切り替える


# ================================================================ データ
def load(path, band, use_cache=True):
    """前処理済みのデータを cache/ に保存して、2 回目からはそれを読む（1 ファイル数十 MB）"""
    cpath = os.path.join("cache", os.path.splitext(os.path.basename(path))[0] + f"_{band}.npz")
    if use_cache and os.path.exists(cpath) and os.path.getmtime(cpath) > os.path.getmtime(path):
        z = np.load(cpath)
        return {k: z[k] for k in z.files}
    d = _load(path, band)
    if use_cache:
        os.makedirs("cache", exist_ok=True)
        np.savez(cpath, **d)
    return d


def _load(path, band):
    emg, F, T, mvc = E.load(path, band)
    Fe, last = E.features(emg)
    X = sosfilt(E.SOS, emg, axis=0).astype(np.float32)
    rest = T.sum(1) == 0
    base = np.median(F[rest], 0) if rest.sum() > 100 else np.median(F, 0)
    Y = ((F - base) / mvc * 100)[last].astype(np.float32)        # 10ch の力 [%MVC]（符号はあとでそろえる）
    act = T.max(1)[last]
    return dict(F=Fe.astype(np.float32), X=X, last=last, Y=Y, act=act)


def sensor_sign(seg):
    """押すと値が増える向きにそろえる（学習データの目標との相関で決める）"""
    out = []
    for i in range(NOUT):
        c = np.corrcoef(seg["Y"][:, i], seg["act"])[0, 1] if seg["Y"][:, i].std() > 0 else 0.0
        out.append(-1.0 if np.isfinite(c) and c < 0 else 1.0)
    return np.array(out, dtype=np.float32)


def view(seg, sc, sign, mask=None):
    n = len(seg["F"])
    m = np.ones(n, bool) if mask is None else mask.copy()
    m[:E.WASHOUT] = False
    X = (seg["X"] / sc["rsd"]).astype(np.float32)
    return dict(F=((seg["F"] - sc["mu"]) / sc["sd"]).astype(np.float32),
                W=sliding_window_view(X, E.WIN, axis=0), widx=seg["last"] - E.WIN + 1,
                Y=seg["Y"] * sign, m=m)


# ================================================================ 深層学習モデル
class LSTMFeat(nn.Module):
    kind = "seq"

    def __init__(s, nin, nch, h, l, drop):
        super().__init__()
        s.rnn = nn.LSTM(nin, h, l, batch_first=True, dropout=drop if l > 1 else 0)
        s.drop, s.out = nn.Dropout(drop), nn.Linear(h, NOUT)

    def forward(s, F, W=None, state=None):
        y, st = s.rnn(F, state)
        return s.out(s.drop(y)), st


class CNNLSTM(nn.Module):
    kind = "seq_raw"

    def __init__(s, nin, nch, c, h, drop):
        super().__init__()
        s.enc = nn.Sequential(
            nn.Conv1d(nch, nch, 15, padding=7, groups=nch, bias=False), nn.BatchNorm1d(nch),   # ch ごとの時間フィルタ
            nn.Conv1d(nch, c, 1, bias=False), nn.BatchNorm1d(c), nn.ELU(),                    # 電極の組み合わせ
            nn.Conv1d(c, c, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.Conv1d(c, c, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(drop))
        s.rnn = nn.LSTM(c, h, 1, batch_first=True)
        s.out = nn.Linear(h, NOUT)

    def embed(s, W):
        B, T = W.shape[:2]
        return s.enc(W.reshape(B * T, *W.shape[2:])).reshape(B, T, -1)

    def forward(s, F, W, state=None):
        y, st = s.rnn(s.embed(W), state)
        return s.out(y), st


class TransformerFeat(nn.Module):
    kind = "ctx"

    def __init__(s, nin, nch, d, l, drop):
        super().__init__()
        s.inp = nn.Linear(nin, d)
        s.pos = nn.Parameter(torch.zeros(1, CTX, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True), l)
        s.out = nn.Linear(d, NOUT)

    def forward(s, Fctx, W=None, state=None):       # Fctx: (B, CTX, nin) → 最後の時刻だけ出力
        z = s.enc(s.inp(Fctx) + s.pos)
        return s.out(z[:, -1]), None


DEEP = {"lstm": LSTMFeat, "cnnlstm": CNNLSTM, "transformer": TransformerFeat}


def nparam(m):
    return sum(p.numel() for p in m.parameters())


_CTX_CACHE = {}


def ctx_windows(F, idx):
    """各時刻 t について、過去 CTX ステップの特徴量（足りない分は 0）"""
    key = (id(F), len(F))
    if key not in _CTX_CACHE:
        P = np.vstack([np.zeros((CTX - 1, F.shape[1]), np.float32), F])
        _CTX_CACHE[key] = (F, sliding_window_view(P, CTX, axis=0).transpose(0, 2, 1))
    return _CTX_CACHE[key][1][idx]


def make_batch(model, segs, rng, B, scale):
    # 学習に使える行が多いデータほど選ばれやすく。系列モデルは SEQ_T より短いデータを使わない（あれば）
    nfit = np.array([s["m"].sum() for s in segs], float)
    if model.kind != "ctx":
        long_enough = np.array([len(s["F"]) >= SEQ_T for s in segs])
        if long_enough.any():
            nfit = nfit * long_enough
    p = nfit / nfit.sum()
    if model.kind == "ctx":                           # 学習に使う行から B 個の時刻を選ぶ
        s = segs[rng.choice(len(segs), p=p)]
        if "fit_idx" not in s:
            s["fit_idx"] = np.flatnonzero(s["m"])
        t = rng.choice(s["fit_idx"], B)
        F = torch.from_numpy(np.ascontiguousarray(ctx_windows(s["F"], t)))
        Y = torch.from_numpy(s["Y"][t]) / torch.from_numpy(scale)
        return F.to(DEV), None, Y.to(DEV), torch.ones(B, dtype=torch.bool, device=DEV)
    out = dict(F=[], W=[], Y=[], M=[])
    T = min(SEQ_T, min(len(s["F"]) for s, w in zip(segs, p) if w > 0))
    for _ in range(B):
        s = segs[rng.choice(len(segs), p=p)]
        a = rng.integers(0, len(s["F"]) - T + 1)
        m = s["m"][a:a + T].copy(); m[:BURN] = False
        out["F"].append(s["F"][a:a + T]); out["Y"].append(s["Y"][a:a + T]); out["M"].append(m)
        if model.kind == "seq_raw":
            out["W"].append(s["W"][s["widx"][a:a + T]])
    t = lambda k: torch.from_numpy(np.stack(out[k]).astype(np.float32 if k != "M" else bool)).to(DEV)
    return t("F"), (t("W") if out["W"] else None), t("Y") / torch.from_numpy(scale).to(DEV), t("M")


@torch.no_grad()
def predict(model, seg, chunk=2048):
    model.eval()
    g = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
    if model.kind == "seq":
        y, _ = model(g(seg["F"])[None])
        return y[0].cpu().numpy()
    if model.kind == "seq_raw":
        e = torch.cat([model.enc(g(seg["W"][seg["widx"][i:i + chunk]]))
                       for i in range(0, len(seg["widx"]), chunk)])
        y, _ = model.rnn(e[None])
        return model.out(y)[0].cpu().numpy()
    n = len(seg["F"])
    return np.vstack([model(g(ctx_windows(seg["F"], np.arange(i, min(i + chunk, n)))))[0].cpu().numpy()
                      for i in range(0, n, chunk)])


def train_deep(model, fit, val, hp, a, rng, scale):
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    n_fit = sum(s["m"].sum() for s in fit)
    bsz = a.batch if model.kind != "ctx" else a.batch * 16
    per = SEQ_T if model.kind != "ctx" else 1
    steps = int(np.clip(n_fit / (bsz * per) * 2, 20, 400))     # 1 エポック ≒ 学習データ 2 周分
    best, best_state, bad, ep = np.inf, copy.deepcopy(model.state_dict()), 0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        for _ in range(steps):
            F, W, Y, M = make_batch(model, fit, rng, bsz, scale)
            if not M.any():
                continue
            y, _ = model(F, W)
            loss = ((y - Y) ** 2)[M].mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vs = [s for s in val if s["m"].any()]                  # 検証用の行を全部つなげて 1 つの NRMSE に
        v = nrmse_all(np.vstack([s["Y"][s["m"]] for s in vs]),
                      np.vstack([predict(model, s)[s["m"]] * scale for s in vs]), scale)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return best, ep


# ================================================================ ESN / Ridge
def ridge(H, Y, lam):
    Z = np.hstack([H, np.ones((len(H), 1))])
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)


def ridge_path(H, Y, lams):
    """いくつもの λ の Ridge を一度に解く（ZᵀZ の固有値分解を 1 回だけ）"""
    Z = np.hstack([H, np.ones((len(H), 1))])
    ev, Q = np.linalg.eigh(Z.T @ Z)
    QtY = Q.T @ (Z.T @ Y)
    return [Q @ (QtY / (ev + lam)[:, None]) for lam in lams]


def lin(Wo, H):
    return np.hstack([H, np.ones((len(H), 1))]) @ Wo


class ESNModel:
    def __init__(s, nin, n, rho, in_scale, leak, seed):
        rng = np.random.default_rng(seed)
        s.Win = (rng.uniform(-1, 1, (n, nin)) * in_scale).astype(np.float64)
        d = min(1.0, 10 / n)                                 # 1 ノードあたり約 10 本の結合
        W = sp.random(n, n, density=d, random_state=np.random.RandomState(seed),
                      data_rvs=lambda k: np.random.RandomState(seed + 1).uniform(-1, 1, k), format="csr")
        if n <= 300:
            ev = np.max(np.abs(np.linalg.eigvals(W.toarray())))
        else:                                                # 大きいときは一番大きい固有値だけ求める
            try:
                ev = np.abs(eigs(W, k=1, which="LM", return_eigenvectors=False, maxiter=5000, tol=1e-4)[0])
            except Exception:                                # 収束しなかったときは理論値で近似
                ev = np.sqrt(n * d / 3)
        W = W * (rho / ev)
        s.W = W.toarray() if n <= 300 else W.tocsr()         # 小さいときは普通の行列の方が速い
        s.leak, s.n = leak, n

    def states(s, F):
        h = np.zeros(s.n); H = np.empty((len(F), s.n))
        Wi = F @ s.Win.T
        for t in range(len(F)):
            h = (1 - s.leak) * h + s.leak * np.tanh(Wi[t] + s.W @ h)
            H[t] = h
        return H

    def nparam(s):
        nnz = s.W.nnz if sp.issparse(s.W) else int((s.W != 0).sum())
        return s.Win.size + nnz + (s.n + 1) * NOUT


def fit_readout(Htr, Ytr, Hv, Yv, scale):
    best = None
    for lam, Wo in zip(LAMS, ridge_path(Htr, Ytr, LAMS)):
        v = nrmse_all(Yv, lin(Wo, Hv), scale)
        if best is None or v < best[0]:
            best = (v, lam, Wo)
    return best


# ================================================================ 評価
def nrmse_all(Y, P, scale):
    return float(np.mean(np.sqrt(((Y - P) ** 2).mean(0)) / scale))


def metrics(Y, P, scale):
    out = []
    for i in range(NOUT):
        y, p = Y[:, i], P[:, i]
        out.append(dict(ch=i + 1, nrmse=np.sqrt(((y - p) ** 2).mean()) / scale[i],
                        r=np.corrcoef(y, p)[0, 1] if y.std() > 0 and p.std() > 0 else np.nan,
                        mae=np.abs(y - p).mean(), lag=E.lag_ms(y, p)))
    return out


def latency_ms(name, obj, nin, nch, reps=200):
    """1 ステップ（20ms ごと 1 回）の推論にかかる時間。1 スレッド"""
    nt = torch.get_num_threads(); torch.set_num_threads(1)
    if name == "ridge":
        x = np.random.randn(nin); f = lambda: lin(obj, x[None])
    elif name == "esn":
        h, x = np.zeros(obj.n), np.random.randn(nin)
        f = lambda: np.tanh(obj.Win @ x + obj.W @ h)
    else:
        obj = copy.deepcopy(obj).cpu().eval()             # 推論時間は Pi 5 を想定して CPU で測る
        if obj.kind == "seq":
            x, st = torch.randn(1, 1, nin), None
            f = lambda: obj(x)
        elif obj.kind == "seq_raw":
            w = torch.randn(1, 1, nch, E.WIN)
            f = lambda: obj(None, w)
        else:
            x = torch.randn(1, CTX, nin)
            f = lambda: obj(x)
    with torch.no_grad():
        for _ in range(20):
            f()
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
    torch.set_num_threads(nt)
    return (time.perf_counter() - t0) / reps * 1000


# ================================================================ メイン
def find_days(root):
    """{被験者: [1 日目 {課題: パス}, 2 日目 {課題: パス}（なければ空）]}"""
    import glob, os
    from collections import defaultdict
    tmp = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(root, "emg_force-*.hdf5")):
        r = E.parse_name(p)
        if r:
            tmp[r[0]][r[2]][r[1]] = p
    return {s: [d[sorted(d)[0]], d[sorted(d)[1]] if len(d) > 1 else {}] for s, d in sorted(tmp.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(E.BANDS), default="all")
    ap.add_argument("--models", nargs="*", default=list(SPEC))
    ap.add_argument("--sizes", nargs="*", default=SIZES)
    ap.add_argument("--subjects", nargs="*", help="省略すると、データがある被験者のうち番号の小さい 3 人")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=4, help="ハイパーパラメータを何通り試すか（全モデル同じ）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--day2", action="store_true", help="2 日目の repeats_short でもテスト（校正なし）")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto = GPU（NVIDIA）があれば使う")
    ap.add_argument("--no_cache", action="store_true", help="前処理の保存（cache/ フォルダ）を使わない")
    ap.add_argument("--tag", default="", help="結果ファイル名に付ける名前（分けて実行するときに上書きを防ぐ）")
    a = ap.parse_args()
    if a.quick:
        a.sizes, a.seeds, a.trials, a.epochs, a.patience = ["S"], [0], 2, 3, 2
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
    global DEV
    if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()):
        DEV = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"GPU を使います: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU で計算します" + ("（GPU が見つかりませんでした）" if a.device == "auto" else ""))

    sess = find_days(a.root)                            # 1 日分だけの被験者も使う（2 日目は --day2 のときだけ）
    need = ["repeats_long", "repeats_short"]
    sess = {k: v for k, v in sess.items() if all(t in v[0] for t in need)}
    subs = [s.zfill(2) for s in a.subjects] if a.subjects else sorted(sess)[:3]
    subs = [s for s in subs if s in sess]
    print(f"被験者 {subs} / {a.band}（{len(E.BANDS[a.band])}ch）/ モデル {a.models} / 大きさ {a.sizes}"
          f" / 乱数の種 {a.seeds} / ハイパーパラメータ {a.trials} 通り" + (" / 2 日目も評価" if a.day2 else "") + "\n")

    rows, t_all = [], time.time()
    for subj in subs:
        d1, d2 = sess[subj]
        t0 = time.time()
        L, S = load(d1["repeats_long"], a.band, not a.no_cache), load(d1["repeats_short"], a.band, not a.no_cache)
        sign = sensor_sign(L)
        sc = dict(mu=L["F"].mean(0), sd=L["F"].std(0) + 1e-8, rsd=L["X"].std(0) + 1e-8)
        fit_m, val_m = split_blocks(len(L["F"]))
        FIT, VAL, TEST = view(L, sc, sign, fit_m), view(L, sc, sign, val_m), view(S, sc, sign)
        Yfit = FIT["Y"][FIT["m"]]
        scale = np.maximum(np.percentile(Yfit, 99, 0) - np.percentile(Yfit, 1, 0), 5.0).astype(np.float32)
        tests = [("同じ日", TEST)]
        if a.day2 and not all(t in d2 for t in need):
            print(f"被験者 {subj}: 2 日目のデータがないので、2 日目の評価はしません")
        if a.day2 and all(t in d2 for t in need):
            L2, S2 = load(d2["repeats_long"], a.band, not a.no_cache), load(d2["repeats_short"], a.band, not a.no_cache)
            sc2 = dict(mu=L2["F"].mean(0), sd=L2["F"].std(0) + 1e-8, rsd=L2["X"].std(0) + 1e-8)
            tests.append(("別の日（校正なし）", view(S2, sc2, sign)))
        nin, nch = L["F"].shape[1], L["X"].shape[1]
        print(f"被験者 {subj}: 読み込み {time.time() - t0:.0f} 秒 / long {len(L['F']) * E.HOP / 60000:.1f} 分"
              f"（学習 {fit_m.sum()} / 検証 {val_m.sum()} ステップ）/ short {len(S['F']) * E.HOP / 60000:.1f} 分")

        hp_rng = np.random.default_rng(1234)              # 試すハイパーパラメータの組は全モデル同じ並び
        hp_deep = [{k: v[hp_rng.integers(len(v))] for k, v in HP_DEEP.items()} for _ in range(a.trials)]
        hp_esn = [{k: v[hp_rng.integers(len(v))] for k, v in HP_ESN.items()} for _ in range(a.trials)]

        def record(name, size, seed, hp, P_of, extra):
            for tname, T in tests:
                k = T["m"]
                for r in metrics(T["Y"][k], P_of(T)[k], scale):
                    rows.append(dict(subj=subj, model=name, size=size, seed=seed, test=tname, hp=str(hp), **r, **extra))

        for name in a.models:
            for size in [s for s in a.sizes if s in SPEC[name]]:
                tn, spec = time.time(), SPEC[name][size]
                if name == "ridge":
                    v, lam, Wo = fit_readout(FIT["F"][FIT["m"]], Yfit, VAL["F"][VAL["m"]], VAL["Y"][VAL["m"]], scale)
                    record(name, size, 0, dict(lam=lam), lambda T: lin(Wo, T["F"]),
                           dict(params=Wo.size, latency_ms=latency_ms(name, Wo, nin, nch), val_nrmse=v, train_sec=time.time() - tn))
                elif name == "esn":
                    # ① ハイパーパラメータ選び（種 0）→ ② 一番良い設定を各種で
                    res = []
                    for hp in hp_esn:
                        m = ESNModel(nin, spec["n"], seed=a.seeds[0], **hp)
                        H = m.states(FIT["F"])
                        res.append((fit_readout(H[FIT["m"]], Yfit, H[VAL["m"]], VAL["Y"][VAL["m"]], scale)[0], hp))
                    v, hp = min(res, key=lambda x: x[0])
                    for seed in a.seeds:
                        ts = time.time()
                        m = ESNModel(nin, spec["n"], seed=seed, **hp)
                        H = m.states(FIT["F"])
                        v, lam, Wo = fit_readout(H[FIT["m"]], Yfit, H[VAL["m"]], VAL["Y"][VAL["m"]], scale)
                        record(name, size, seed, dict(hp, lam=lam), lambda T: lin(Wo, m.states(T["F"])),
                               dict(params=m.nparam(), latency_ms=latency_ms(name, m, nin, nch), val_nrmse=v,
                                    train_sec=time.time() - ts))
                else:
                    res = []
                    for hp in hp_deep:
                        torch.manual_seed(a.seeds[0])
                        m = DEEP[name](nin, nch, **spec, drop=hp["drop"]).to(DEV)
                        v, _ = train_deep(m, [FIT], [VAL], hp, a, np.random.default_rng(a.seeds[0]), scale)
                        res.append((v, hp))
                    v, hp = min(res, key=lambda x: x[0])
                    for seed in a.seeds:
                        ts = time.time(); torch.manual_seed(seed)
                        m = DEEP[name](nin, nch, **spec, drop=hp["drop"]).to(DEV)
                        v, ep = train_deep(m, [FIT], [VAL], hp, a, np.random.default_rng(seed), scale)
                        record(name, size, seed, hp, lambda T: predict(m, T) * scale,
                               dict(params=nparam(m), latency_ms=latency_ms(name, m, nin, nch), val_nrmse=v,
                                    epochs=ep, train_sec=time.time() - ts))
                last = pd.DataFrame([r for r in rows if r["subj"] == subj and r["model"] == name
                                     and r["size"] == size and r["test"] == "同じ日"])
                print(f"  {name:11s} {size:2s}: テスト NRMSE {last.nrmse.mean():.3f}  r {last.r.mean():.2f}"
                      f"  （パラメータ {int(last.params.iloc[0]):,} / 推論 {last.latency_ms.mean():.3f} ms"
                      f" / {time.time() - tn:.0f} 秒）", flush=True)

    # ================================================================ まとめ
    df = pd.DataFrame(rows)
    tag = f"personal_{a.band}" + (f"_{a.tag}" if a.tag else "")
    df.to_csv(tag + ".csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    order = [(m, s) for m in a.models for s in SIZES if s in a.sizes and s in SPEC[m]]
    for tname in df.test.unique():
        d = df[df.test == tname]
        # 被験者×種ごとに 10ch を平均 → その平均とばらつき
        per = d.groupby(["model", "size", "subj", "seed"])[["nrmse", "r", "mae", "lag"]].mean()
        g = per.groupby(["model", "size"])
        tab = pd.concat([g.nrmse.mean().rename("NRMSE"), g.nrmse.std().rename("NRMSEのばらつき"),
                         g.r.mean().rename("r"), g.mae.mean().rename("MAE"), g.lag.mean().rename("遅れms"),
                         d.groupby(["model", "size"]).params.first().rename("パラメータ数"),
                         d.groupby(["model", "size"]).latency_ms.mean().rename("推論ms"),
                         d.groupby(["model", "size"]).train_sec.mean().rename("学習秒")], axis=1)
        tab = tab.reindex([o for o in order if o in tab.index])
        print(f"\n[{tname}：モデル × 大きさ（{len(d.subj.unique())} 人 × 種 {len(a.seeds)} の平均。NRMSE は低いほど良い）]")
        print(tab.round({"NRMSE": 3, "NRMSEのばらつき": 3, "r": 2, "MAE": 1, "遅れms": 0, "推論ms": 3, "学習秒": 0}).to_string())
    print("\n[被験者ごとの NRMSE（同じ日、種で平均）]")
    print(df[df.test == "同じ日"].pivot_table(index=["model", "size"], columns="subj", values="nrmse", aggfunc="mean")
          .reindex([o for o in order]).round(3).to_string())
    print("\n[センサーごとの NRMSE（同じ日、全モデルの最良サイズ）]")
    best = df[df.test == "同じ日"].groupby(["model", "size"]).nrmse.mean().groupby("model").idxmin()
    sub = df[(df.test == "同じ日") & df.set_index(["model", "size"]).index.isin(best.values)]
    print(sub.pivot_table(index="model", columns="ch", values="nrmse", aggfunc="mean").round(3).to_string())

    # グラフ：横軸 パラメータ数 / 推論時間、縦軸 NRMSE
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = dict(ridge="#555555", esn="#d9622b", lstm="#2b7bd9", cnnlstm="#2ba84a", transformer="#8a4bd9")
    d = df[df.test == "同じ日"]
    per = d.groupby(["model", "size", "subj", "seed"]).agg(nrmse=("nrmse", "mean"), params=("params", "first"),
                                                          lat=("latency_ms", "mean")).reset_index()
    for mname in a.models:
        g = per[per.model == mname].groupby("size").agg(n=("nrmse", "mean"), sd=("nrmse", "std"),
                                                         p=("params", "first"), l=("lat", "mean"))
        g = g.reindex([s for s in SIZES if s in g.index])
        for k, xcol, xl in [(0, "p", "パラメータ数"), (1, "l", "1 ステップの推論時間 [ms]（このPC・1 スレッド）")]:
            ax[k].errorbar(g[xcol], g.n, yerr=g.sd.fillna(0), fmt="o-", color=colors.get(mname), label=mname, capsize=3)
            for s, row in g.iterrows():
                ax[k].annotate(s, (row[xcol], row.n), textcoords="offset points", xytext=(4, 4), fontsize=8)
            ax[k].set_xscale("log"); ax[k].set_xlabel(xl); ax[k].grid(alpha=0.3)
    ax[0].set_ylabel("NRMSE（同じ日のテスト、低いほど良い）"); ax[0].legend()
    fig.suptitle(f"モデルの大きさと精度（被験者 {', '.join(subs)}、種 {len(a.seeds)} 個、エラーバー = 標準偏差）")
    plt.tight_layout(); plt.savefig(tag + ".png", dpi=130)
    print(f"\n保存しました: {tag}.csv / {tag}.png（全体 {(time.time() - t_all) / 60:.1f} 分）")


if __name__ == "__main__":
    main()
