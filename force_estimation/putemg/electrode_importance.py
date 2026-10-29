"""
【PC 側】どの電極が力の推定に効いているかを調べる（電極ごとの重要度）。
export フォルダ（model_*.npz / .onnx / replay_*.npz）を使う。データのダウンロードや GPU は不要。

調べ方（並べ替え重要度 = permutation importance）
  1. テストデータ（1 日目 repeats_short）でふつうに推定し、誤差 NRMSE を測る（基準）
  2. 電極 1 つぶんの特徴量（log RMS と log WL）だけ、時間の順番をバラバラに並べ替える
     → その電極の情報だけが「意味のない値」になる。ほかの電極はそのまま
  3. もう一度推定して、誤差が基準から何 % 増えたかを見る
     → 大きく増える電極ほど、モデルがその電極に頼っている
  これを 24 電極すべて、さらにバンド（8 電極）まとめても行う。
  誤差は「力 10ch 全体」「親指（センサー1・2）」「人差し指（センサー3・4）」の 3 つで別々に測る。

使い方（export フォルダがある場所で）:
  pip install onnxruntime matplotlib scipy        ← 初回だけ
  python electrode_importance.py export/model_transformer_S_03.npz
  python electrode_importance.py export/model_transformer_S_03.npz --repeats 3     # 並べ替えを 3 回して平均（より安定・3 倍時間）
  python electrode_importance.py export/model_ridge_03.npz                         # ほかのモデルでも可
出力: importance_<モデル>.png（図）と importance_<モデル>.csv（数値）
"""
import argparse, json, os, time
import numpy as np
from scipy.signal import sosfilt

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--map", help="センサーと指の対応 json（省略時: 親指=1,2 / 人差し指=3,4）")
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

B = dict(np.load(a.model, allow_pickle=True))
kind = str(B["kind"])
name = os.path.basename(a.model).replace("model_", "").replace(".npz", "")
tag = name + ("" if str(B.get("band", "all")) == "all" else "_" + str(B["band"]))
subj = name.rsplit("_", 1)[-1]
R = np.load(os.path.join(os.path.dirname(a.model), f"replay_{subj}.npz"))
emg, Y, mask = R["emg"].astype(np.float64), R["Y"].astype(np.float64), R["mask"]
SOS, WIN, HOP, CTX = B["sos"], int(B["win"]), int(B["hop"]), int(B["ctx"])
mu, sd, scale = B["mu"], B["sd"], B["scale"]
nch = emg.shape[1]

FING = {"thumb": [0, 1], "index": [2, 3]}
if a.map:
    m = json.load(open(a.map, encoding="utf-8"))["sensor_to_finger"]
    FING = {f: [int(k) - 1 for k, v in m.items() if v == f] for f in ["thumb", "index"]}
TARGETS = [("全体（10ch）", list(range(Y.shape[1]))), ("親指", FING["thumb"]), ("人差し指", FING["index"])]

# ---------------------------------------------------------------- 特徴量（pi_replay.py と同じ計算をまとめて行う）
x = sosfilt(SOS, emg, axis=0)
n = min(len(Y), (len(x) - WIN) // HOP + 1)
idx = np.arange(n)[:, None] * HOP + np.arange(WIN)[None, :]
w = x[idx]                                                    # (時間, 150, 電極)
rms = np.log(np.sqrt((w ** 2).mean(1)) + 1e-6)
wl = np.log(np.abs(np.diff(w, axis=1)).sum(1) + 1e-6)
F0 = ((np.concatenate([rms, wl], 1) - mu) / sd).astype(np.float32)
Y, mask = Y[:n], mask[:n]

# ---------------------------------------------------------------- 推論（時系列まるごと）
if kind in ("transformer", "lstm"):
    import onnxruntime as ort
    sess = ort.InferenceSession(a.model.replace(".npz", ".onnx"), providers=["CPUExecutionProvider"])


def run(F):
    if kind == "transformer":
        Fp = np.concatenate([np.zeros((CTX - 1, F.shape[1]), np.float32), F])
        win = np.lib.stride_tricks.sliding_window_view(Fp, (CTX, F.shape[1]))[:, 0]   # (時間, CTX, 特徴量)
        return np.array([sess.run(None, {"x": win[i:i + 1]})[0][0] for i in range(len(F))]) * scale
    if kind == "lstm":
        h = np.zeros((int(B["layers"]), 1, int(B["hidden"])), np.float32); c = h.copy(); out = []
        for f in F:
            y, h, c = sess.run(None, {"x": f[None, None, :], "h": h, "c": c}); out.append(y[0])
        return np.array(out) * scale
    if kind == "esn":
        Win, W, leak, Wo = B["Win"], B["W"], float(B["leak"]), B["Wo"]
        h = np.zeros(W.shape[0]); H = np.empty((len(F), W.shape[0]))
        for i, f in enumerate(F.astype(np.float64)):
            h = (1 - leak) * h + leak * np.tanh(Win @ f + W @ h); H[i] = h
        return np.c_[H, np.ones(len(F))] @ Wo
    return np.c_[F, np.ones(len(F))] @ B["Wo"]


def nrmse(P):
    e = np.sqrt(((Y[mask] - P[mask]) ** 2).mean(0)) / scale        # センサーごと
    return np.array([e[ids].mean() for _, ids in TARGETS])


t0 = time.time()
P0 = run(F0)
base = nrmse(P0)
sec = time.time() - t0
nb = max(1, nch // 8)
BANDNAME = {"all": ["肘側", "中央", "手首側"], "elbow_middle": ["肘側", "中央"], "middle_wrist": ["中央", "手首側"],
            "elbow_wrist": ["肘側", "手首側"], "elbow": ["肘側"], "middle": ["中央"], "wrist": ["手首側"]}
FIRST = {"肘側": 1, "中央": 9, "手首側": 17}                    # バンドの最初の電極番号（putEMG の番号）
BAND = BANDNAME.get(str(B.get("band", "all" if nch == 24 else "elbow_wrist")))
NUM = [FIRST[b] + p for b in BAND for p in range(8)]           # 入力の何番目 → putEMG の電極番号
groups = [(f"電極{NUM[e]}", [e]) for e in range(nch)] + [(f"{BAND[j]}バンド全体", list(range(8 * j, 8 * j + 8))) for j in range(nb)]
print(f"モデル {name}（{kind}）/ {nch}ch / テスト {mask.sum()} ステップ")
print("基準の NRMSE: " + " / ".join(f"{t} {v:.3f}" for (t, _), v in zip(TARGETS, base)))
print(f"1 回の推定 {sec:.0f} 秒 × {len(groups) * a.repeats} 回 → 約 {sec * len(groups) * a.repeats / 60:.1f} 分かかります")

rng = np.random.default_rng(a.seed)
imp = np.zeros((len(groups), len(TARGETS)))
for gi, (g, es) in enumerate(groups):
    cols = es + [nch + e for e in es]                               # その電極の RMS と WL
    acc = []
    for _ in range(a.repeats):
        F = F0.copy(); F[:, cols] = F0[rng.permutation(n)][:, cols]
        acc.append(nrmse(run(F)) / base - 1)
    imp[gi] = np.mean(acc, 0) * 100
    print(f"  {g:10s} 誤差の増え方: " + " / ".join(f"{t} {v:+5.1f}%" for (t, _), v in zip(TARGETS, imp[gi])), flush=True)

# ---------------------------------------------------------------- 保存（CSV と図）
with open(f"importance_{tag}.csv", "w", encoding="utf-8-sig") as fo:
    fo.write("電極," + ",".join(f"{t} 誤差の増加[%]" for t, _ in TARGETS) + "\n")
    for (g, _), v in zip(groups, imp):
        fo.write(g + "," + ",".join(f"{x:.2f}" for x in v) + "\n")

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
E = imp[:nch]
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
vmax = max(E.max(), 1e-6)
for j, (t, _) in enumerate(TARGETS):
    ax = fig.add_subplot(gs[0, j])
    grid = E[:, j].reshape(nb, 8)
    im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=E[:, j].max() + 1e-6, aspect="auto")
    for b in range(nb):
        for p in range(8):
            v = grid[b, p]
            ax.text(p, b, f"{NUM[8 * b + p]}\n{v:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 * grid.max() else "black")
    ax.set_yticks(range(nb)); ax.set_yticklabels(BAND[:nb]); ax.set_xticks([])
    ax.set_xlabel("腕の周りの並び（上の数字 = 電極番号）")
    ax.set_title(f"{t}　基準 NRMSE {base[j]:.3f}", fontsize=11)
ax = fig.add_subplot(gs[1, :])
xs = np.arange(1, nch + 1); wd = 0.27
cols = ["#555555", "#d9622b", "#2ba84a"]
for j, (t, _) in enumerate(TARGETS):
    ax.bar(xs + (j - 1) * wd, E[:, j], wd, color=cols[j], label=t)
for j in range(1, nb):
    ax.axvline(8 * j + 0.5, color="gray", lw=0.8, ls=":")
for j in range(nb):
    ax.text(8 * j + 4.5, ax.get_ylim()[1] * 0.95, BAND[j] + "バンド", ha="center", va="top", fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels(NUM); ax.set_xlabel("電極の番号"); ax.set_ylabel("並べ替えたときの誤差の増加 [%]")
ax.axhline(0, color="black", lw=0.6); ax.grid(alpha=0.3, axis="y"); ax.legend(loc="upper right")
bt = "  ".join(f"{g}: " + " / ".join(f"{v:.0f}%" for v in imp[nch + j]) for j, (g, _) in enumerate(groups[nch:]))
fig.suptitle(f"電極ごとの重要度（{tag}）― 大きいほどモデルがその電極に頼っている\n"
             f"バンドまとめて並べ替え（全体 / 親指 / 人差し指）→ {bt}", fontsize=11)
plt.tight_layout()
out = f"importance_{tag}.png"
plt.savefig(out, dpi=120)
print(f"\n保存しました: {out} / importance_{tag}.csv")
