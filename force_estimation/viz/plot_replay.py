"""
【PC 側】再生データ（録画した筋電）と、正解の力・モデルの予測をグラフにする（静止画）。
export_model.py が書き出した export フォルダを使う。ラズパイは不要。

  上段 : 筋電の波形（3 ch、フィルタ後）
  2 段 : 24ch（16ch）の筋電の大きさを色で表したもの（縦：電極、横：時間）
  3 段 : 力センサー 10ch の平均 [%MVC] … 正解と予測
  4 段 : 選んだセンサー 2 つ … 正解と予測

使い方:
  python plot_replay.py export/model_transformer_S_03.npz
  python plot_replay.py export/model_transformer_S_03.npz --t0 30 --t1 60     # 30〜60 秒だけ
  python plot_replay.py export/model_transformer_S_03.npz --chs 2 7 --save replay.png
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--t0", type=float, default=0)
ap.add_argument("--t1", type=float, default=None)
ap.add_argument("--chs", type=int, nargs=2, help="下段に出すセンサー番号（省略すると、力の変化が大きい 2 つ）")
ap.add_argument("--map", help="finger_map.py が作った finger_map_XX.json。付けると下段に 親指 と 人差し指 を表示")
ap.add_argument("--save")
a = ap.parse_args()
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]

B = np.load(a.model, allow_pickle=True)
subj = os.path.basename(a.model).rsplit("_", 1)[-1].split(".")[0]
R = np.load(os.path.join(os.path.dirname(a.model), f"replay_{subj}.npz"))
emg, Y, P, k = R["emg"], R["Y"], B["pred_pc"], R["mask"]
fs, win, hop = int(B["fs"]), int(B["win"]), int(B["hop"])
x = sosfilt(B["sos"], emg, axis=0)
te = np.arange(len(x)) / fs
tf = (win + np.arange(len(Y)) * hop) / fs
t1 = a.t1 if a.t1 is not None else te[-1]
me, mf = (te >= a.t0) & (te <= t1), (tf >= a.t0) & (tf <= t1)

# 50ms ごとの RMS（色の図用）
step = 50
n = len(x) // step
rms = np.sqrt((x[:n * step].reshape(n, step, -1) ** 2).mean(1))
tr = np.arange(n) * step / fs
mr = (tr >= a.t0) & (tr <= t1)

kk = k & mf
r_all = np.corrcoef(Y[kk].mean(1), P[kk].mean(1))[0, 1]
c1, c2 = a.chs if a.chs else (np.argsort(Y.std(0))[::-1][:2] + 1)
fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.3, 1.3, 1.3]})
nch = x.shape[1]
show = [0, nch // 2, nch - 1]
for j, c in enumerate(show):
    ax[0].plot(te[me][::2], x[me, c][::2] + j * 4 * x[:, show].std(), lw=0.4, label=f"ch{c + 1}")
ax[0].set_ylabel("筋電の波形"); ax[0].set_yticks([]); ax[0].legend(loc="upper right", fontsize=8, ncol=3)
im = ax[1].imshow(np.log(rms[mr].T + 1e-6), aspect="auto", origin="lower", cmap="magma",
                  extent=[tr[mr][0], tr[mr][-1], 0.5, nch + 0.5])
ax[1].set_ylabel(f"筋電の大きさ\n（電極 1〜{nch}）")
ax[2].plot(tf[mf], Y[mf].mean(1), color="black", lw=1.4, label="正解（力センサー）")
ax[2].plot(tf[mf], P[mf].mean(1), color="#2b7bd9", lw=1.4, label="予測")
ax[2].set_ylabel("力 10ch の平均\n[%MVC]"); ax[2].legend(loc="upper right", fontsize=9)
ax[2].set_title(f"表示範囲の相関 r = {r_all:.2f}", fontsize=10, loc="left")
if a.map:                                                      # 親指・人差し指に当たるセンサーの平均
    import json
    mp = json.load(open(a.map, encoding="utf-8"))["sensor_to_finger"]
    curves = [(f, [int(k) - 1 for k, v in mp.items() if v == key], col)
              for f, key, col in [("親指", "thumb", "#d9622b"), ("人差し指", "index", "#2ba84a")]]
else:
    curves = [(f"センサー{c}", [c - 1], col) for c, col in [(c1, "#d9622b"), (c2, "#2ba84a")]]
for name, ids, col in curves:
    ax[3].plot(tf[mf], Y[mf][:, ids].mean(1), color=col, lw=1.3, label=f"{name} 正解")
    ax[3].plot(tf[mf], P[mf][:, ids].mean(1), color=col, lw=1.3, ls="--", label=f"{name} 予測")
ax[3].set_ylabel("力 [%MVC]"); ax[3].legend(loc="upper right", fontsize=8, ncol=2)
ax[3].set_xlabel("時間 [秒]")
for x_ in ax[2:]:
    x_.grid(alpha=0.3)
fig.suptitle(f"再生データと予測（{os.path.basename(a.model)}）")
plt.tight_layout()
if a.save:
    plt.savefig(a.save, dpi=120); print("保存しました:", a.save)
else:
    plt.show()
