"""
putEMG-Force で ESN（2 段構え）を評価する。被験者ごとにモデルを作り、
  ・同じ日の別の課題（セッション内）
  ・別の日（セッション間：電極を付け直した後）
の両方で精度を比べる。

データ（putemg_force フォルダ）:
  emg_force-<被験者>-<課題>-<日時>.hdf5
  課題: bias / mvc / repeats_long / sequential / repeats_short。日付の早い方を 1 回目のセッションとする。

分け方（被験者ごと）:
  学習         : 1 回目の repeats_long ＋ sequential
  校正         : 1 回目の repeats_short の前半（接触判定のしきい値を決める）
  評価(同じ日) : 1 回目の repeats_short の後半
  別の日の準備 : 2 回目の repeats_long の最初の 60 秒（付けた直後を想定。--adapt/--align/--recal で使う）
  評価(別の日) : 2 回目の sequential ＋ repeats_short

使い方（putemg-downloader フォルダで）:
  pip install pandas tables h5py numpy scipy matplotlib
  python esn_putemg.py                      # 前腕中央のバンド(8ch)、4 つの指グループ
  python esn_putemg.py --band all           # 24ch すべて
  python esn_putemg.py --band wrist         # 手首側のバンド
  python esn_putemg.py --band middle_wrist  # 16ch：中央＋手首側（elbow_middle / elbow_wrist も可）
  python esn_putemg.py --target thumb       # 親指だけ推定
  python esn_putemg.py --adapt              # 別の日：付けた直後の筋電で標準化し直す（力の正解は不要）
  python esn_putemg.py --align              # 別の日：電極バンドの回転を自動で合わせる（力の正解は不要）
  python esn_putemg.py --recal              # 別の日：付けた直後 60 秒の力の正解で読み出しを学習し直す
  python esn_putemg.py --rotinv             # 回転に左右されない特徴を使う（バンドの向きを気にせず付けられる）
  python esn_putemg.py --rotate_test 2 --align   # 検証：バンドを 90° 回して付けた場合を再現し、回転合わせで戻せるか
  python esn_putemg.py --plot               # 別の日の評価データの一部をグラフ表示

力の単位: その被験者・セッションの MVC（FORCE_MVC）に対する % [%MVC]
"""
import argparse, glob, os, re, sys, time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, resample_poly, sosfilt, tf2sos

# ---------------------------------------------------------------- 設定
FS_RAW = 5120
FS = 1000                   # 5120Hz → 1000Hz に変換（PiEEG-16 で 1kHz 記録する想定に合わせる）
WIN, HOP = 150, 20          # 窓 150ms、20ms ごとに推定
NOTCH = 50                  # ポーランド(50Hz)。函館も 50Hz
BANDS = {"elbow": range(1, 9), "middle": range(9, 17), "wrist": range(17, 25), "all": range(1, 25),
         # 16ch（2 本のバンド）。PiEEG-16 と同じ ch 数
         "elbow_middle": list(range(1, 17)), "middle_wrist": list(range(9, 25)),
         "elbow_wrist": list(range(1, 9)) + list(range(17, 25))}
GROUPS = ["thumb", "index", "middle", "ringsmall"]   # TRAJ_1..4 の順
WASHOUT = 25
# 判定のしきい値 [%MVC]
CONTACT, REST, STRONG, DROP_Y, DROP_P = 3.0, 1.5, 20.0, 8.0, 3.0

SOS = np.vstack([butter(4, [20, 450], btype="band", fs=FS, output="sos"),
                 tf2sos(*iirnotch(NOTCH, 30, FS))])


# ---------------------------------------------------------------- 読み込み
def parse_name(path):
    m = re.match(r"emg_force-(\d+)-([a-z_]+)-(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return (m.group(1), m.group(2), m.group(3)) if m else None


def find_sessions(root):
    """{被験者: [ {課題: パス}(1回目), {課題: パス}(2回目) ]}"""
    tmp = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(root, "emg_force-*.hdf5")):
        r = parse_name(p)
        if r:
            subj, traj, date = r
            tmp[subj][date][traj] = p
    out = {}
    for subj, dates in sorted(tmp.items()):
        ds = sorted(dates)
        if len(ds) >= 2:
            out[subj] = [dates[ds[0]], dates[ds[-1]]]
        else:
            print(f"被験者 {subj}: セッションが 1 回分しかないのでスキップ")
    return out


def load(path, band):
    emg_cols = [f"EMG_{i}" for i in BANDS[band]]
    cols = emg_cols + [f"FORCE_{i}" for i in range(1, 11)] + ["FORCE_MVC"] + [f"TRAJ_{i}" for i in range(1, 5)]
    df = pd.read_hdf(path, "data", columns=cols)
    raw = np.ascontiguousarray(df[emg_cols].to_numpy(np.float64))   # 連続した配列にしないと resample_poly が NaN を出すことがある
    emg = resample_poly(raw, 25, 128, axis=0)                        # 5120 → 1000Hz
    if not np.isfinite(emg).all():
        sys.exit(f"{os.path.basename(path)}: 筋電に NaN/inf が含まれています")
    idx = np.round(np.arange(len(emg)) * FS_RAW / FS).astype(int).clip(0, len(df) - 1)
    F = df[[f"FORCE_{i}" for i in range(1, 11)]].to_numpy(np.float64)[idx]
    T = df[[f"TRAJ_{i}" for i in range(1, 5)]].to_numpy(np.float64)[idx]
    mvc = float(df["FORCE_MVC"].iloc[0])
    return emg, F, T, mvc


def finger_forces(F, T, mvc):
    """10 個の力センサーを 4 つの指グループにまとめ、%MVC にする。
    ・ゼロ点: 目標(TRAJ)がすべて 0 の区間の中央値を引く
    ・どのセンサーがどの指か: 目標の軌道との相関が一番高いグループに割り当てる（符号も相関から決める）"""
    rest = T.sum(1) == 0
    F = F - (np.median(F[rest], 0) if rest.sum() > 100 else np.median(F, 0))
    G = np.zeros((len(F), 4))
    for i in range(10):
        r = [np.corrcoef(F[:, i], T[:, g])[0, 1] if T[:, g].std() > 0 else 0 for g in range(4)]
        g = int(np.nanargmax(np.abs(r)))
        G[:, g] += np.sign(r[g]) * F[:, i]
    return G / mvc * 100.0


def features(emg):
    x = sosfilt(SOS, emg, axis=0)
    idx = np.arange(WIN, len(x) + 1, HOP)
    n = x.shape[1]
    Fe = np.empty((len(idx), 2 * n))
    for k, i in enumerate(idx):
        seg = x[i - WIN:i]
        Fe[k, :n] = np.log(np.sqrt((seg ** 2).mean(0)) + 1e-6)
        Fe[k, n:] = np.log(np.abs(np.diff(seg, axis=0)).sum(0) + 1e-6)
    return Fe, idx - 1


def prepare(path, band, target):
    emg, F, T, mvc = load(path, band)
    Fe, last = features(emg)
    Y = finger_forces(F, T, mvc)[last]
    if target != "fingers":
        Y = Y[:, [GROUPS.index(target)]]
    return Fe, Y


# ---------------------------------------------------------------- ESN と 2 段構え（esn_force.py と同じ考え方）
class ESN:
    def __init__(self, n_in, n_res=300, rho=0.9, leak=1.0, in_scale=0.3, density=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, (n_res, n_in)) * in_scale
        W = rng.uniform(-1, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < density)
        self.W = W * rho / np.max(np.abs(np.linalg.eigvals(W)))
        self.leak, self.h = leak, np.zeros(n_res)

    def run(self, U):
        self.h[:] = 0
        H = np.empty((len(U), len(self.h)))
        for t, u in enumerate(U):
            self.h = (1 - self.leak) * self.h + self.leak * np.tanh(self.Win @ u + self.W @ self.h)
            H[t] = self.h
        return H


def smooth(P, a):
    S = np.array(P, dtype=float)
    for i in range(1, len(S)):
        S[i] = a * S[i] + (1 - a) * S[i - 1]
    return S


def ridge_fit(H, Y, lam):
    Z = np.hstack([H, np.ones((len(H), 1))])
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)


def apply(Wo, H):
    return np.hstack([H, np.ones((len(H), 1))]) @ Wo


class TwoStage:
    GATE_SMOOTH, MIN_ON, MIN_OFF, RAMP = 0.15, 10, 5, 0.2

    def __init__(self, lam):
        self.lam, self.on, self.off = lam, 0.6, 0.4

    def fit(self, H, Y):
        C = (Y.max(1, keepdims=True) > CONTACT).astype(float)
        self.Wc = ridge_fit(H, C, self.lam)
        on = C[:, 0] > 0
        self.Wf = ridge_fit(H[on], Y[on], self.lam)
        self.Wb = ridge_fit(H, Y, self.lam)

    def gate(self, c):
        g = np.zeros(len(c)); state, count, level = False, 0, 0.0
        for t, v in enumerate(c):
            count += 1
            if not state and v > self.on and count >= self.MIN_OFF:
                state, count = True, 0
            elif state and v < self.off and count >= self.MIN_ON:
                state, count = False, 0
            level = min(1.0, level + self.RAMP) if state else max(0.0, level - self.RAMP)
            g[t] = level
        return g

    def predict(self, H, a, two_stage=True):
        if not two_stage:
            return np.clip(smooth(apply(self.Wb, H), a), 0, None)
        c = smooth(apply(self.Wc, H), self.GATE_SMOOTH)[:, 0]
        return np.clip(smooth(apply(self.Wf, H), a), 0, None) * self.gate(c)[:, None]

    def tune(self, H, Y, a):
        best = None
        for on in np.arange(0.3, 0.81, 0.05):
            for gap in [0.0, 0.1, 0.2, 0.3]:
                self.on, self.off = on, on - gap
                P = self.predict(H, a)
                s = np.abs(Y - P).mean() + 2 * dropout(Y.max(1), P.max(1)) * 10
                if best is None or s < best[0]:
                    best = (s, on, on - gap)
        _, self.on, self.off = best


# ---------------------------------------------------------------- 評価
def lag_ms(y, p, maxl=40):
    y, p = y - y.mean(), p - p.mean()
    return int(np.argmax([np.dot(y[:len(y) - l], p[l:]) for l in range(maxl)])) * HOP


def dropout(y, p):
    m = y > DROP_Y
    return (p[m] < DROP_P).mean() if m.any() else 0.0


def metrics(y, p):
    rest, strong = y < REST, y > STRONG
    rise = (p[1:] >= CONTACT) & (p[:-1] < CONTACT) & rest[1:]
    return dict(r=np.corrcoef(y, p)[0, 1] if p.std() > 0 and y.std() > 0 else np.nan,
                mae=np.abs(y - p).mean(), lag=lag_ms(y, p),
                rest=p[rest].mean() if rest.any() else np.nan,
                ratio=p[strong].mean() / y[strong].mean() if strong.any() else np.nan,
                drop=dropout(y, p), fs=rise.sum() / (len(y) * HOP / 1000 / 60))


# ---------------------------------------------------------------- メイン
def channel_perm(band, k, flip):
    """各バンド（8 電極・45°間隔）の中で、電極を k 個ぶん回転（flip なら向きも反転）した並びを返す"""
    chans = list(BANDS[band]); perm = []
    for c in chans:
        b0 = (c - 1) // 8 * 8; j = (c - 1) % 8
        j2 = ((-j if flip else j) + k) % 8
        perm.append(chans.index(b0 + j2 + 1))
    return np.array(perm)


def permute_features(F, perm):
    n = len(perm)
    return np.hstack([F[:, :n][:, perm], F[:, n:][:, perm]])


def rotinv(F):
    """回転に左右されない特徴に変換する。
    各バンドの 8 電極の値の並び（前腕を一周する並び）を空間方向にフーリエ変換し、その大きさだけを使う。
    並びをぐるっと回しても（向きを反転しても）大きさは変わらないので、バンドをどの向きに付けても同じ値になる。
    8 電極 → 5 個（周波数 0〜4）。log RMS と log WL のそれぞれに適用。"""
    n = F.shape[1] // 2
    out = []
    for half in (F[:, :n], F[:, n:]):
        X = half.reshape(len(F), n // 8, 8)
        out.append(np.abs(np.fft.rfft(X, axis=2)).reshape(len(F), -1))
    return np.hstack(out)


def signature(F, n):
    """電極の並びを表す特徴：各電極の平均的な活動の大きさ ＋ 電極どうしの活動の連動（相関行列）"""
    R = F[:, :n]
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    C = np.corrcoef(R.T)[np.triu_indices(n, 1)]
    return np.concatenate([z(R.mean(0)), z(C)])


def best_rotation(F_ref, F_new, band, margin=0.15):
    """1 回目（学習データ）と新しい日で、電極の並びの特徴が一番そろう回転を探す（力の正解は使わない）。
    回転しない場合より margin 以上よくならなければ、回転しない（迷ったら動かさない）。"""
    n = len(BANDS[band]); ref = signature(F_ref, n)
    scores = {}
    for flip in [False, True]:
        for k in range(8):
            p = signature(permute_features(F_new, channel_perm(band, k, flip)), n)
            scores[(k, flip)] = np.corrcoef(ref, p)[0, 1]
    (k, flip), r = max(scores.items(), key=lambda kv: kv[1])
    r0 = scores[(0, False)]
    if r - r0 < margin:
        return r0, 0, False
    return r, k, flip


def run_subject(subj, sess, a, results, plot_store):
    s1, s2 = sess
    need = ["repeats_long", "sequential", "repeats_short"]
    if not all(k in s1 for k in need) or not all(k in s2 for k in need):
        print(f"被験者 {subj}: 必要なファイルがそろっていないのでスキップ"); return
    t0 = time.time()
    # ---- 1 回目：学習・校正・同じ日の評価
    tr = [prepare(s1[k], a.band, a.target) for k in ["repeats_long", "sequential"]]
    Fs, Ys = prepare(s1["repeats_short"], a.band, a.target)
    h = len(Fs) // 2
    cal, te1 = (Fs[:h], Ys[:h]), (Fs[h:], Ys[h:])
    # ---- 2 回目：最初の cal_sec 秒（repeats_long の冒頭）＝付けた直後の準備用、評価は sequential と repeats_short
    Fl, Yl = prepare(s2["repeats_long"], a.band, a.target)
    nst = int(a.cal_sec * 1000 / HOP)
    start = (Fl[:nst], Yl[:nst])
    te2 = [prepare(s2[k], a.band, a.target) for k in ["sequential", "repeats_short"]]
    if a.rotate_test:   # 検証用：2 回目のバンドをわざと回して付けたことにする
        rp = channel_perm(a.band, a.rotate_test, False)
        start = (permute_features(start[0], rp), start[1])
        te2 = [(permute_features(F, rp), y) for F, y in te2]

    if a.rotinv:   # 回転に左右されない特徴に変換（--align は不要になる）
        tf = lambda pairs: [(rotinv(F), y) for F, y in pairs]
        tr, te2 = tf(tr), tf(te2)
        cal, te1, start = tf([cal])[0], tf([te1])[0], tf([start])[0]
    Ftr = np.vstack([F for F, _ in tr])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    esn = ESN(Ftr.shape[1], a.n_res, a.rho, a.leak, a.in_scale)

    def states(pairs, m, s, perm=None):
        H, Y = [], []
        for F, y in pairs:
            if perm is not None:
                F = permute_features(F, perm)
            H.append(esn.run((F - m) / s)[WASHOUT:]); Y.append(y[WASHOUT:])
        return np.vstack(H), np.vstack(Y)

    Htr, Ytr = states(tr, mu, sd)
    model = TwoStage(a.ridge); model.fit(Htr, Ytr)
    Hc, Yc = states([cal], mu, sd); model.tune(Hc, Yc, a.smooth)
    H1, Y1 = states([te1], mu, sd)

    # ---- 別の日の準備（付けた直後の cal_sec 秒だけを使う）
    perm, info = None, ""
    if a.align and not a.rotinv:
        r, k, flip = best_rotation(Ftr, start[0], a.band)
        perm = channel_perm(a.band, k, flip)
        info += f" / 回転 {k * 45}°{'（反転）' if flip else ''}（一致度 {r:.2f}）"
    Fst = permute_features(start[0], perm) if perm is not None else start[0]
    mu2, sd2 = (Fst.mean(0), Fst.std(0) + 1e-8) if a.adapt else (mu, sd)
    if a.recal:   # 付けた直後の cal_sec 秒を「力の正解付き」で使い、読み出しとしきい値を学習し直す
        Hs, Ys2 = states([start], mu2, sd2, perm)
        m2 = TwoStage(a.ridge); m2.fit(np.vstack([Htr, Hs]), np.vstack([Ytr, Ys2]))
        m2.tune(Hs, Ys2, a.smooth)
        info += f" / 再校正後のしきい値 ON {m2.on:.2f}・OFF {m2.off:.2f}"
    else:
        m2 = model
    H2, Y2 = states(te2, mu2, sd2, perm)

    names = GROUPS if a.target == "fingers" else [a.target]
    for tag, mdl, H, Y in [("同じ日", model, H1, Y1), ("別の日", m2, H2, Y2)]:
        P = mdl.predict(H, a.smooth)
        for j, n in enumerate(names):
            results.append(dict(subj=subj, eval=tag, out=n, **metrics(Y[:, j], P[:, j])))
        if tag == "別の日" and subj not in plot_store:
            plot_store[subj] = (Y, P, names)
    print(f"被験者 {subj}: 完了（{time.time() - t0:.0f} 秒）"
          f"  しきい値 ON {model.on:.2f} / OFF {model.off:.2f}{info}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(BANDS), default="middle")
    ap.add_argument("--target", choices=["fingers"] + GROUPS, default="fingers")
    ap.add_argument("--n_res", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--in_scale", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--adapt", action="store_true", help="別の日：付けた直後の筋電で標準化し直す（正解不要）")
    ap.add_argument("--align", action="store_true", help="別の日：電極バンドの回転を自動で合わせる（正解不要）")
    ap.add_argument("--rotinv", action="store_true", help="回転に左右されない特徴（電極の並びの空間フーリエ変換の大きさ）を使う")
    ap.add_argument("--recal", action="store_true", help="別の日：付けた直後の力の正解付きデータで読み出しを学習し直す")
    ap.add_argument("--rotate_test", type=int, default=0,
                    help="検証用：2 回目のバンドを 45°×この数だけ回して付けたことにする（--align が戻せるかを確かめる）")
    ap.add_argument("--cal_sec", type=float, default=60, help="付けた直後の準備に使う秒数（初期値 60 秒）")
    ap.add_argument("--subjects", nargs="*", help="例: --subjects 03 04")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    sessions = find_sessions(a.root)
    if a.subjects:
        sessions = {k: v for k, v in sessions.items() if k in [s.zfill(2) for s in a.subjects]}
    if not sessions:
        sys.exit(f"{a.root} に 2 回分そろった被験者のデータが見つかりません。")
    print(f"被験者 {list(sessions)} / バンド {a.band}（{len(BANDS[a.band])}ch）/ 対象 {a.target}"
          f"\n別の日の準備: 標準化 {'あり' if a.adapt else 'なし'} / 回転合わせ {'あり' if a.align else 'なし'}"
          f" / 再校正 {'あり' if a.recal else 'なし'}（最初の {a.cal_sec:g} 秒）"
          f" / 回転に左右されない特徴 {'あり' if a.rotinv else 'なし'}\n")

    results, plot_store = [], {}
    for subj, sess in sessions.items():
        run_subject(subj, sess, a, results, plot_store)
    if not results:
        sys.exit("評価できた被験者がいませんでした。")

    df = pd.DataFrame(results)
    cols = ["r", "mae", "lag", "rest", "ratio", "drop", "fs"]
    head = {"r": "r", "mae": "MAE[%MVC]", "lag": "遅れ[ms]", "rest": "非接触時の誤出力[%MVC]",
            "ratio": "強い力の再現率", "drop": "途切れ率", "fs": "誤った立ち上がり[回/分]"}
    pd.set_option("display.width", 200)
    print("\n[被験者ごとの結果]")
    print(df.set_index(["subj", "eval", "out"])[cols].rename(columns=head).round(2).to_string())
    print("\n[全被験者の平均]")
    print(df.groupby(["eval", "out"])[cols].mean().rename(columns=head).round(2).to_string())

    if a.plot:
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
        subj, (Y, P, names) = next(iter(plot_store.items()))
        n = min(3000, len(Y)); tt = np.arange(n) * HOP / 1000
        fig, ax = plt.subplots(len(names), 1, figsize=(12, 2 + 2 * len(names)), sharex=True, squeeze=False)
        for j, nm in enumerate(names):
            ax[j, 0].plot(tt, Y[:n, j], "k", lw=1.1, label="実測")
            ax[j, 0].plot(tt, P[:n, j], color="#d9622b", lw=1.0, label="ESN 2 段構え")
            ax[j, 0].set_ylabel(f"{nm} [%MVC]")
        ax[0, 0].legend(loc="upper right"); ax[0, 0].set_title(f"被験者 {subj}：別の日のデータ")
        ax[-1, 0].set_xlabel("time [s]")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
