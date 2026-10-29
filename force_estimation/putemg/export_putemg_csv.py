"""
putEMG-Force の hdf5 を CSV（Excel で開ける）に書き出す。esn_putemg.py と同じフォルダに置いて使う。

出力する列
  time_s                 : 時刻 [秒]
  EMG_1 … EMG_24         : 筋電（--band で選んだ ch だけ）
  thumb, index, middle, ringsmall : 4 つの指グループの力 [%MVC]（学習の「正解」に使っているもの）
  TRAJ_1 … TRAJ_4        : 画面に出した目標の力 [%MVC]
  FORCE_1 … FORCE_10     : 力センサーの生の値（--sensors を付けたときだけ）

--rate で細かさを選ぶ
  1000 : 1000Hz（初期値。学習で使っているのと同じ）
  raw  : 元の 5120Hz（ファイルがとても大きくなる）
  feat : 20ms ごとの特徴量（log RMS / log WL）と指の力。ESN に入れているものそのもの。一番小さい

使い方（putemg-downloader フォルダで）:
  python export_putemg_csv.py putemg_force\\emg_force-03-repeats_short-2018-05-11-11-34-34-187.hdf5
  python export_putemg_csv.py putemg_force                         # フォルダ内を全部
  python export_putemg_csv.py <ファイル> --t0 0 --t1 60             # 最初の 60 秒だけ
  python export_putemg_csv.py <ファイル> --band elbow_wrist --rate feat
  python export_putemg_csv.py <ファイル> --sensors                  # 力センサーの生の値も

※ 書き出す前に大きさを見積もり、ディスクの空きが足りなければ止まります。
"""
import argparse, glob, os, shutil, sys
import numpy as np
import pandas as pd

import esn_putemg as E

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+", help="hdf5 ファイル、またはそれが入ったフォルダ")
ap.add_argument("--band", choices=list(E.BANDS), default="all")
ap.add_argument("--rate", choices=["1000", "raw", "feat"], default="1000")
ap.add_argument("--t0", type=float, default=0)
ap.add_argument("--t1", type=float, default=None)
ap.add_argument("--sensors", action="store_true", help="力センサー 10 個の生の値も出す")
ap.add_argument("--out", default="csv_out")
ap.add_argument("--yes", action="store_true", help="大きくても確認せずに書き出す")
a = ap.parse_args()

files = []
for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, "*.hdf5"))) if os.path.isdir(p) else [p]
if not files:
    sys.exit("hdf5 ファイルが見つかりません")
os.makedirs(a.out, exist_ok=True)
ch = list(E.BANDS[a.band])


def build(path):
    """1 ファイル分の表（DataFrame）を作る"""
    if a.rate == "feat":
        Fe, Y = E.prepare(path, a.band, "fingers")
        n = len(ch)
        t = (E.WIN - 1 + np.arange(len(Fe)) * E.HOP) / E.FS
        cols = {"time_s": t}
        cols.update({f"logRMS_{c}": Fe[:, i] for i, c in enumerate(ch)})
        cols.update({f"logWL_{c}": Fe[:, n + i] for i, c in enumerate(ch)})
        cols.update({g: Y[:, j] for j, g in enumerate(E.GROUPS)})
        return pd.DataFrame(cols)

    if a.rate == "1000":
        emg, F, T, mvc = E.load(path, a.band)
        fs = E.FS
    else:   # raw 5120Hz
        cols = [f"EMG_{i}" for i in ch] + [f"FORCE_{i}" for i in range(1, 11)] + ["FORCE_MVC"] + \
               [f"TRAJ_{i}" for i in range(1, 5)]
        df = pd.read_hdf(path, "data", columns=cols)
        emg = df[[f"EMG_{i}" for i in ch]].to_numpy(np.float64)
        F = df[[f"FORCE_{i}" for i in range(1, 11)]].to_numpy(np.float64)
        T = df[[f"TRAJ_{i}" for i in range(1, 5)]].to_numpy(np.float64)
        mvc = float(df["FORCE_MVC"].iloc[0])
        fs = E.FS_RAW
    G = E.finger_forces(F, T, mvc)
    out = {"time_s": np.arange(len(emg)) / fs}
    out.update({f"EMG_{c}": emg[:, i] for i, c in enumerate(ch)})
    out.update({g: G[:, j] for j, g in enumerate(E.GROUPS)})
    out.update({f"TRAJ_{j + 1}": T[:, j] / mvc * 100 for j in range(4)})
    if a.sensors:
        out.update({f"FORCE_{i + 1}": F[:, i] for i in range(10)})
    return pd.DataFrame(out)


for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    print(f"読み込み中: {name}")
    df = build(path)
    t1 = a.t1 if a.t1 is not None else df.time_s.iloc[-1]
    df = df[(df.time_s >= a.t0) & (df.time_s <= t1)]

    est = len(df) * df.shape[1] * 11                    # 1 つの数値 ≒ 11 文字として見積もる
    free = shutil.disk_usage(a.out).free
    print(f"  {len(df):,} 行 × {df.shape[1]} 列, 長さ {df.time_s.iloc[-1] - df.time_s.iloc[0]:.0f} 秒 "
          f"→ 約 {est / 1e6:.0f} MB（ディスクの空き {free / 1e6:.0f} MB）")
    if est > free * 0.8:
        print("  ※ 空きが足りないのでスキップします。--t0/--t1 で短くするか --rate feat を使ってください")
        continue
    if est > 200e6 and not a.yes:
        if input("  200MB を超えます。書き出しますか？ [y/N] ").strip().lower() != "y":
            continue
    if len(df) > 1_048_575:
        print("  ※ 104 万行を超えるので、Excel では途中までしか表示されません（Python などでは全部読めます）")

    suffix = f"_{a.band}_{a.rate}" + (f"_{a.t0:g}-{t1:g}s" if a.t0 > 0 or a.t1 is not None else "")
    out = os.path.join(a.out, name + suffix + ".csv")
    df.to_csv(out, index=False, float_format="%.6g", encoding="utf-8-sig")
    print(f"  保存しました: {out}（{os.path.getsize(out) / 1e6:.1f} MB）")
