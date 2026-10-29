"""
校正実験：2 日目に付け直したあと、短い校正（30 秒〜数分）で、どのモデルがどれだけ精度を取り戻せるか。
personal_compare.py・esn_putemg.py・esn_calcurve.py と同じフォルダに置いて使う。

■ 流れ（被験者ごと）
  1 日目 : repeats_long で学習（10 秒ごとのかたまりで 学習 4 : 検証 1）。ハイパーパラメータは --trials 通りから選ぶ
  2 日目 : sequential（全部の指を順に）→ repeats_long の先頭 N 秒を「校正データ」（同じく 学習 4 : 検証 1 に分ける）
           テストは 2 日目の repeats_short（全部）。校正の長さを変えても同じものを使う
  参考   : 1 日目の repeats_short でのテスト（同じ日の精度）

■ 校正の方法
  なし       : 1 日目のモデルのまま（筋電の標準化だけ校正データでし直す。力の正解は使わない）
  一から     : 校正データだけで新しく学習（教授のやり方。ESN は読み出しを Ridge で学習し直す）
  全体を微調整: 1 日目のモデルを出発点に、全部の重みを小さい学習率で学習
  出力層だけ : 最後の層だけ学習し直す（ESN の読み出し再学習に相当。一番軽い）
  入力層だけ : 入力の直後に置いた小さな線形層（最初は何もしない）だけ学習 → 電極のずれを入力側で直す
  ※ ESN は「なし / 一から / 前の重みに寄せる（Ridge の正則化を 1 日目の重みに向ける）」の 3 つ

■ モデル（大きさは personal_compare の結果から、各モデルで良かったもの）
  esn M / lstm S / cnnlstm L / transformer S     （--pick lstm=M のように変えられる）

使い方（emg_compare フォルダで）:
  python calib_compare.py --quick                       # 動作確認（1 人・短く）
  python calib_compare.py                               # 本番（3 人、種 2 つ）
  python calib_compare.py --band elbow_wrist --tag ch16 # 16ch（PiEEG-16 と同じ）
  python calib_compare.py --secs 30 60 120 240 99999    # 校正の長さ（秒。99999 = 使えるだけ全部）

結果: calib_<band>[_<tag>].csv / .png
"""
import argparse, copy, time, types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import esn_putemg as E
import personal_compare as P
from esn_calcurve import split_blocks

PICK = {"esn": "M", "lstm": "S", "cnnlstm": "L", "transformer": "S"}
CNN_ONLY = ["CNN固定", "CNN固定+BN更新", "CNN固定+混ぜる層"]                     # cnnlstm だけで試す方法
METHODS_DEEP = ["なし", "一から", "全体を微調整", "出力層だけ", "入力層だけ"] + CNN_ONLY
METHODS_ESN = ["なし", "一から", "前の重みに寄せる"]


# ================================================================ 入力アダプタ付きのモデル
class FeatAdapter(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.lin = nn.Linear(n, n)
        nn.init.eye_(s.lin.weight); nn.init.zeros_(s.lin.bias)

    def forward(s, x):
        return s.lin(x)


class RawAdapter(FeatAdapter):
    def forward(s, w):                      # w: (B, ch, 150) → ch を混ぜ直す
        return s.lin(w.transpose(1, 2)).transpose(1, 2)


class Adapted(nn.Module):
    """personal_compare のモデルの前に入力アダプタを付けたもの（predict などはそのまま使える形）"""
    def __init__(s, base, nin, nch):
        super().__init__()
        s.base, s.kind = base, base.kind
        if s.kind == "seq_raw":
            s.adp = RawAdapter(nch)
            s.enc = nn.Sequential(s.adp, base.enc)
            s.rnn, s.out = base.rnn, base.out
        else:
            s.adp = FeatAdapter(nin)
        s.head = base.out

    def forward(s, F, W=None, state=None):
        if s.kind == "seq_raw":
            B, T = W.shape[:2]
            e = s.enc(W.reshape(B * T, *W.shape[2:])).reshape(B, T, -1)
            y, st = s.rnn(e, state)
            return s.out(y), st
        return s.base(s.adp(F), W, state)


def make_deep(name, size, nin, nch, drop):
    return Adapted(P.DEEP[name](nin, nch, **P.SPEC[name][size], drop=drop), nin, nch).to(P.DEV)


def _train_keep_cnn_eval(self, mode=True):
    """CNN固定: 学習中も CNN（BatchNorm を含む）は推論モードのまま"""
    nn.Module.train(self, mode)
    self.enc.eval()
    return self


@torch.no_grad()
def update_bn(m, fit, chunk=2048):
    """CNN の BatchNorm の平均・分散だけを、校正データで計算し直す（学習なし、順伝播 1 回）"""
    bns = [x for x in m.enc.modules() if isinstance(x, nn.BatchNorm1d)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None              # None = 全データの単純平均
    m.enc.train()
    for s in fit:
        idx = s["widx"][s["m"]] if s["m"].any() else s["widx"]
        for i in range(0, len(idx), chunk):
            m.enc(torch.from_numpy(np.ascontiguousarray(s["W"][idx[i:i + chunk]])).to(P.DEV))
    for b in bns:
        b.momentum = 0.1
    m.enc.eval()


def finetune(model, which, fit, val, hp, a, rng, scale):
    m = copy.deepcopy(model)
    if which in CNN_ONLY:                    # CNN（入力アダプタ含む）は基本固定、LSTM と出力層を学習
        if which == "CNN固定+BN更新":
            update_bn(m, fit)
        params = list(m.rnn.parameters()) + list(m.out.parameters())
        if which == "CNN固定+混ぜる層":       # 電極を混ぜ合わせる 1×1 畳み込み（16 → 64）も学習
            params += list(m.base.enc[2].parameters())
        m.train = types.MethodType(_train_keep_cnn_eval, m)
    else:
        params = {"全体を微調整": list(m.parameters()), "出力層だけ": list(m.head.parameters()),
                  "入力層だけ": list(m.adp.parameters())}[which]
    ids = {id(p) for p in params}
    for p in m.parameters():
        p.requires_grad_(id(p) in ids)
    lr = hp["lr"] * (0.3 if which == "全体を微調整" else 1.0)
    v, ep = P.train_deep(m, fit, val, dict(hp, lr=lr), a, rng, scale)
    for p in m.parameters():
        p.requires_grad_(True)
    if which in CNN_ONLY:
        del m.train
    return m, ep


# ================================================================ ESN
def ridge_prior_path(H, Y, lams, W0):
    """(ZᵀZ + λI) W = ZᵀY + λ W0 をいくつもの λ で解く（λ が大きいほど 1 日目の重みに近い）"""
    Z = np.hstack([H, np.ones((len(H), 1))])
    ev, Q = np.linalg.eigh(Z.T @ Z)
    ZtY = Z.T @ Y
    return [Q @ ((Q.T @ (ZtY + lam * W0)) / (ev + lam)[:, None]) for lam in lams]


def pick_best(Ws, Hv, Yv, scale):
    vs = [P.nrmse_all(Yv, P.lin(W, Hv), scale) for W in Ws]
    i = int(np.argmin(vs))
    return Ws[i], i


# ================================================================ データ
def cut(seg, k):
    k = min(k, len(seg["F"]))
    return dict(F=seg["F"][:k], X=seg["X"][:seg["last"][k - 1] + 1], last=seg["last"][:k],
                Y=seg["Y"][:k], act=seg["act"][:k])


def scaler(segs):
    F = np.vstack([s["F"] for s in segs]); X = np.vstack([s["X"] for s in segs])
    return dict(mu=F.mean(0), sd=F.std(0) + 1e-8, rsd=X.std(0) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(E.BANDS), default="all")
    ap.add_argument("--models", nargs="*", default=list(PICK))
    ap.add_argument("--pick", nargs="*", default=[], help="大きさを変える 例: lstm=M transformer=M")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--trials", type=int, default=3, help="1 日目のハイパーパラメータを何通り試すか")
    ap.add_argument("--secs", type=float, nargs="*", default=[30, 60, 120, 240, 99999])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--methods", nargs="*", help="試す校正の方法をしぼる（例: 一から 全体を微調整 CNN固定）。なし は常に計算")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.trials, a.epochs, a.patience, a.secs = [0], 1, 3, 2, [60, 240]
        a.subjects = a.subjects or ["03"]
    for kv in a.pick:
        k, v = kv.split("="); PICK[k] = v
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
    if a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available()):
        P.DEV = torch.device("cuda"); torch.backends.cudnn.benchmark = True
        print(f"GPU を使います: {torch.cuda.get_device_name(0)}")
    else:
        P.DEV = torch.device("cpu"); print("CPU で計算します")

    sess = P.find_days(a.root)
    need1, need2 = ["repeats_long", "repeats_short"], ["sequential", "repeats_long", "repeats_short"]
    sess = {k: v for k, v in sess.items() if all(t in v[0] for t in need1) and all(t in v[1] for t in need2)}
    subs = [s.zfill(2) for s in a.subjects] if a.subjects else sorted(sess)[:3]
    subs = [s for s in subs if s in sess]
    print(f"被験者 {subs} / {a.band}（{len(E.BANDS[a.band])}ch）/ モデル "
          f"{[m + ' ' + PICK[m] for m in a.models]} / 種 {a.seeds} / 校正 {a.secs} 秒\n")

    rows, t_all = [], time.time()
    for subj in subs:
        d1, d2 = sess[subj]
        t0 = time.time()
        ld = lambda p: P.load(p, a.band)
        L1, S1 = ld(d1["repeats_long"]), ld(d1["repeats_short"])
        Q2, L2, S2 = ld(d2["sequential"]), ld(d2["repeats_long"]), ld(d2["repeats_short"])
        sign = P.sensor_sign(L1)
        sc1 = scaler([L1])
        fm, vm = split_blocks(len(L1["F"]))
        FIT1, VAL1, TEST1 = P.view(L1, sc1, sign, fm), P.view(L1, sc1, sign, vm), P.view(S1, sc1, sign)
        Yf = FIT1["Y"][FIT1["m"]]
        scale = np.maximum(np.percentile(Yf, 99, 0) - np.percentile(Yf, 1, 0), 5.0).astype(np.float32)
        nin, nch = L1["F"].shape[1], L1["X"].shape[1]

        # 校正データ（長さごと）：sequential → repeats_long の先頭から
        total = len(Q2["F"]) + len(L2["F"])
        cal_sets = []
        for k in sorted(set(min(int(x * 1000 / E.HOP), total) for x in a.secs)):
            parts = [cut(Q2, k)] + ([cut(L2, k - len(Q2["F"]))] if k > len(Q2["F"]) else [])
            sc = scaler(parts)
            n = sum(len(p["F"]) for p in parts)
            f, v = split_blocks(n)
            fit, val, i = [], [], 0
            for p in parts:
                kk = len(p["F"])
                fit.append(P.view(p, sc, sign, f[i:i + kk])); val.append(P.view(p, sc, sign, v[i:i + kk])); i += kk
            cal_sets.append(dict(min=round(n * E.HOP / 60000, 2), fit=fit, val=val, test=P.view(S2, sc, sign)))
        print(f"被験者 {subj}: 読み込み {time.time() - t0:.0f} 秒 / 校正に使える最大 {cal_sets[-1]['min']:.1f} 分")

        def rec(model, method, cal_min, seed, P_of, T, extra):
            k = T["m"]
            for r in P.metrics(T["Y"][k], P_of(T)[k], scale):
                rows.append(dict(subj=subj, model=model, method=method, cal_min=cal_min, seed=seed, **r, **extra))

        hp_rng = np.random.default_rng(1234)
        hp_deep = [{k: v[hp_rng.integers(len(v))] for k, v in P.HP_DEEP.items()} for _ in range(a.trials)]
        hp_esn = [{k: v[hp_rng.integers(len(v))] for k, v in P.HP_ESN.items()} for _ in range(a.trials)]

        for name in a.models:
            size, tn = PICK[name], time.time()
            label = f"{name} {size}"
            if name == "esn":
                n = P.SPEC["esn"][size]["n"]
                res = []
                for hp in hp_esn:
                    m = P.ESNModel(nin, n, seed=a.seeds[0], **hp); H = m.states(FIT1["F"])
                    res.append((P.fit_readout(H[FIT1["m"]], Yf, H[VAL1["m"]], VAL1["Y"][VAL1["m"]], scale)[0], hp))
                hp = min(res, key=lambda x: x[0])[1]
                for seed in a.seeds:
                    m = P.ESNModel(nin, n, seed=seed, **hp); H = m.states(FIT1["F"])
                    _, lam0, W0 = P.fit_readout(H[FIT1["m"]], Yf, H[VAL1["m"]], VAL1["Y"][VAL1["m"]], scale)
                    rec(label, "同じ日（参考）", np.nan, seed, lambda T: P.lin(W0, m.states(T["F"])), TEST1, {})
                    for cs in cal_sets:
                        Hs = [m.states(s["F"]) for s in cs["fit"]]
                        Hf = np.vstack([H[s["m"]] for H, s in zip(Hs, cs["fit"])])
                        Yc = np.vstack([s["Y"][s["m"]] for s in cs["fit"]])
                        Hv = np.vstack([H[s["m"]] for H, s in zip(Hs, cs["val"])])
                        Yv = np.vstack([s["Y"][s["m"]] for s in cs["val"]])
                        Hte = m.states(cs["test"]["F"])
                        rec(label, "なし", cs["min"], seed, lambda T: P.lin(W0, Hte), cs["test"], dict(cal_sec=0))
                        tc = time.time(); _, _, Wc = P.fit_readout(Hf, Yc, Hv, Yv, scale)
                        rec(label, "一から", cs["min"], seed, lambda T: P.lin(Wc, Hte), cs["test"],
                            dict(cal_sec=time.time() - tc))
                        tc = time.time(); Wp, _ = pick_best(ridge_prior_path(Hf, Yc, P.LAMS + [1e4, 1e5], W0), Hv, Yv, scale)
                        rec(label, "前の重みに寄せる", cs["min"], seed, lambda T: P.lin(Wp, Hte), cs["test"],
                            dict(cal_sec=time.time() - tc))
            else:
                res = []
                for hp in hp_deep:
                    torch.manual_seed(a.seeds[0])
                    m = make_deep(name, size, nin, nch, hp["drop"])
                    v, _ = P.train_deep(m, [FIT1], [VAL1], hp, a, np.random.default_rng(a.seeds[0]), scale)
                    res.append((v, hp))
                hp = min(res, key=lambda x: x[0])[1]
                for seed in a.seeds:
                    torch.manual_seed(seed); rng = np.random.default_rng(seed)
                    m1 = make_deep(name, size, nin, nch, hp["drop"])
                    P.train_deep(m1, [FIT1], [VAL1], hp, a, rng, scale)
                    pred = lambda mm: (lambda T: P.predict(mm, T) * scale)
                    rec(label, "同じ日（参考）", np.nan, seed, pred(m1), TEST1, {})
                    for cs in cal_sets:
                        rec(label, "なし", cs["min"], seed, pred(m1), cs["test"], dict(cal_sec=0))
                        for meth in [x for x in METHODS_DEEP[1:] if (x not in CNN_ONLY or name == "cnnlstm")
                                     and (not a.methods or x in a.methods)]:
                            tc = time.time()
                            if meth == "一から":
                                torch.manual_seed(seed)
                                mm = make_deep(name, size, nin, nch, hp["drop"])
                                _, ep = P.train_deep(mm, cs["fit"], cs["val"], hp, a, rng, scale)
                            else:
                                mm, ep = finetune(m1, meth, cs["fit"], cs["val"], hp, a, rng, scale)
                            rec(label, meth, cs["min"], seed, pred(mm), cs["test"],
                                dict(cal_sec=time.time() - tc, epochs=ep))
            d = pd.DataFrame([r for r in rows if r["subj"] == subj and r["model"] == label])
            best = d[d.cal_min == d.cal_min.max()].groupby("method").nrmse.mean()
            print(f"  {label:14s}: 同じ日 {d[d.method == '同じ日（参考）'].nrmse.mean():.3f} / 最長の校正で "
                  + " ".join(f"{k} {v:.3f}" for k, v in best.items()) + f"  （{time.time() - tn:.0f} 秒）", flush=True)

    # ================================================================ まとめ
    df = pd.DataFrame(rows)
    tag = f"calib_{a.band}" + (f"_{a.tag}" if a.tag else "")
    df.to_csv(tag + ".csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    per = df.groupby(["model", "method", "cal_min", "subj", "seed"], dropna=False)[["nrmse", "r", "lag"]].mean().reset_index()
    ref = per[per.method == "同じ日（参考）"].groupby("model").nrmse.mean()
    cal = per[per.method != "同じ日（参考）"].copy()
    mx = cal.groupby("subj").cal_min.transform("max")
    cal["len"] = np.where((cal.cal_min == mx) & (max(a.secs) >= 99999), "最大", cal.cal_min.map(lambda v: f"{v:g}分"))
    order = sorted(cal["len"].unique(), key=lambda v: 1e9 if v == "最大" else float(v[:-1]))
    models = [f"{m} {PICK[m]}" for m in a.models]

    print("\n[同じ日（参考）の NRMSE]")
    print(ref.reindex(models).round(3).to_string())
    for key, lab, dg in [("nrmse", "NRMSE（低いほど良い）", 3), ("r", "相関 r", 2)]:
        print(f"\n[2 日目・校正後の {lab}：行 = モデル・方法、列 = 校正の長さ（{len(subs)} 人 × 種 {len(a.seeds)} の平均）]")
        t = cal.pivot_table(index=["model", "method"], columns="len", values=key, aggfunc="mean")[order]
        print(t.reindex([(m, x) for m in models for x in (METHODS_ESN if m.startswith("esn") else METHODS_DEEP)
                         if (m, x) in t.index]).round(dg).to_string())
    print("\n[各モデルで一番良い方法（長さごと）と、教授のやり方（ESN 一から）との比較]")
    best = cal.groupby(["model", "method", "len"]).nrmse.mean().reset_index()
    bb = best.loc[best.groupby(["model", "len"]).nrmse.idxmin()].pivot(index="model", columns="len", values="nrmse")[order]
    bm = best.loc[best.groupby(["model", "len"]).nrmse.idxmin()].pivot(index="model", columns="len", values="method")[order]
    print(bb.round(3).to_string()); print(bm.to_string())
    print("\n[校正の計算時間（秒、このPC）]")
    print(df[df.method.isin(METHODS_DEEP[1:] + METHODS_ESN[1:])].groupby(["model", "method"]).cal_sec.mean().round(1).to_string())
    print("\n[被験者ごと：最長の校正で一番良い方法の NRMSE]")
    last = cal[cal["len"] == order[-1]].groupby(["model", "method", "subj"]).nrmse.mean().reset_index()
    print(last.loc[last.groupby(["model", "subj"]).nrmse.idxmin()].pivot(index="model", columns="subj", values="nrmse").round(3).to_string())

    # グラフ：モデルごとに 1 枚（方法ごとの線）＋ 各モデルの最良を 1 枚に
    fig, ax = plt.subplots(1, len(models) + 1, figsize=(4.3 * (len(models) + 1), 4.2), sharey=True)
    col = {"なし": "#999999", "一から": "#d9622b", "全体を微調整": "#2b7bd9", "出力層だけ": "#2ba84a",
           "入力層だけ": "#8a4bd9", "前の重みに寄せる": "#2b7bd9", "CNN固定": "#c0392b", "CNN固定+BN更新": "#e67e22", "CNN固定+混ぜる層": "#16a085"}
    mcol = ["#d9622b", "#2b7bd9", "#2ba84a", "#8a4bd9", "#555555"]
    for k, mname in enumerate(models):
        d = cal[cal.model == mname]
        for meth, dd in d.groupby("method"):
            g = dd.groupby("cal_min").nrmse.mean()
            ax[k].plot(g.index, g.values, "o-", color=col.get(meth, "k"), label=meth, ms=4)
        ax[k].axhline(ref.get(mname, np.nan), color="k", ls="--", lw=1, label="同じ日")
        ax[k].set_title(mname); ax[k].set_xscale("log"); ax[k].grid(alpha=0.3); ax[k].legend(fontsize=7)
        ax[k].set_xlabel("校正時間 [分]")
        g = d.groupby(["cal_min", "method"]).nrmse.mean().groupby("cal_min").min()
        ax[-1].plot(g.index, g.values, "o-", color=mcol[k % len(mcol)], label=mname)
    ax[-1].set_title("各モデルの一番良い方法"); ax[-1].set_xscale("log"); ax[-1].grid(alpha=0.3)
    ax[-1].legend(fontsize=8); ax[-1].set_xlabel("校正時間 [分]")
    ax[0].set_ylabel("NRMSE（2 日目、低いほど良い）")
    plt.tight_layout(); plt.savefig(tag + ".png", dpi=130)
    print(f"\n保存しました: {tag}.csv / {tag}.png（全体 {(time.time() - t_all) / 60:.1f} 分）")


if __name__ == "__main__":
    main()
