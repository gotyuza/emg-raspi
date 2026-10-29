"""
putEMG-Force のファイルを 1 つグラフにする（esn_putemg.py と同じフォルダに置いて使う）。

  上段 : 画面に表示された目標の力（TRAJ）
  中段 : 実際に測った指の力（esn_putemg.py の学習で「正解」として使っているもの）[%MVC]
  下段 : 24ch の筋電の大きさ（RMS）を色で表したもの（縦：電極、横：時間）

使い方:
  python plot_putemg.py putemg_force\\emg_force-03-sequential-2018-05-11-11-28-55-137.hdf5
  python plot_putemg.py <ファイル> --t0 30 --t1 90      # 30〜90 秒だけ表示
  python plot_putemg.py <ファイル> --save fig.png        # 画像として保存
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt

import esn_putemg as E

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--t0", type=float, default=0)
ap.add_argument("--t1", type=float, default=None)
ap.add_argument("--save", default=None)
a = ap.parse_args()

plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]

emg, F, T, mvc = E.load(a.path, "all")              # 1000Hz にそろえた 24ch の筋電・力センサー・目標
G = E.finger_forces(F, T, mvc)                      # 4 つの指グループの力 [%MVC]
Tp = T / mvc * 100                                  # 目標も %MVC に
t = np.arange(len(emg)) / E.FS

# 筋電の大きさ：フィルタ → 50ms ごとの RMS
x = sosfilt(E.SOS, emg, axis=0)
step = 50
n = len(x) // step
rms = np.sqrt((x[:n * step].reshape(n, step, -1) ** 2).mean(1))
trms = np.arange(n) * step / E.FS

t1 = a.t1 if a.t1 is not None else t[-1]
m = (t >= a.t0) & (t <= t1)
mr = (trms >= a.t0) & (trms <= t1)
d = 10                                              # 線のグラフは 10 点ごとに間引いて軽くする

names = ["親指", "人差し指", "中指", "薬指＋小指"]
colors = ["#d9622b", "#2b7bd9", "#2ba84a", "#8a4bd9"]
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                       gridspec_kw={"height_ratios": [1, 1.3, 1.6]})

for j in range(4):
    ax[0].plot(t[m][::d], Tp[m, j][::d], color=colors[j], lw=1.2, label=names[j])
    ax[1].plot(t[m][::d], G[m, j][::d], color=colors[j], lw=1.0, label=names[j])
ax[0].set_ylabel("目標の力 [%MVC]")
ax[1].set_ylabel("測った力 [%MVC]")
ax[0].legend(loc="upper right", ncol=4, fontsize=9)
ax[0].set_title(os.path.basename(a.path))

im = ax[2].imshow(np.log(rms[mr].T + 1e-6), aspect="auto", origin="lower", cmap="magma",
                  extent=[trms[mr][0], trms[mr][-1], 0.5, 24.5])
ax[2].set_yticks([4.5, 12.5, 20.5])
ax[2].set_yticklabels(["肘側\n1-8", "中央\n9-16", "手首側\n17-24"])
for y in (8.5, 16.5):
    ax[2].axhline(y, color="white", lw=0.8)
ax[2].set_ylabel("筋電の大きさ（電極）")
ax[2].set_xlabel("時間 [s]")
fig.colorbar(im, ax=ax[2], pad=0.01, label="log RMS")

plt.tight_layout()
if a.save:
    plt.savefig(a.save, dpi=130)
    print("保存しました:", a.save)
else:
    plt.show()
