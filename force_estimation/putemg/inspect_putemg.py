"""
putEMG-Force の hdf5 ファイルの中身（構造・列名・値の範囲）を表示する。

使い方（putemg-downloader フォルダで）:
  pip install pandas tables h5py
  python inspect_putemg.py putemg_force\\emg_force-03-repeats_short-2018-05-11-11-34-34-187.hdf5
"""
import sys
import h5py
import numpy as np

path = sys.argv[1]
print("=== ファイル:", path)

# 1) hdf5 の階層構造
print("\n[hdf5 の構造]")
with h5py.File(path, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:60s} {obj.shape} {obj.dtype}")
    f.visititems(show)

# 2) pandas で読めるか（putEMG は pandas の形式で保存されている）
try:
    import pandas as pd
    with pd.HDFStore(path, "r") as st:
        keys = st.keys()
    print("\n[pandas のキー]", keys)
    for k in keys:
        df = pd.read_hdf(path, k)
        print(f"\n=== キー {k}: {df.shape[0]} 行 × {df.shape[1]} 列")
        print("列名:", list(df.columns))
        print("\n先頭 3 行:")
        print(df.head(3).to_string())
        print("\n数値の列の範囲:")
        num = df.select_dtypes(include=[np.number])
        print(num.describe().T[["mean", "std", "min", "max"]].to_string())
        cat = df.select_dtypes(exclude=[np.number])
        for c in cat.columns:
            u = pd.unique(df[c])
            print(f"\n文字/カテゴリの列 '{c}': {len(u)} 種類 → {list(u)[:15]}")
        # 時刻の列らしきものがあれば、サンプリング間隔を推定
        for c in df.columns:
            if "time" in str(c).lower() or "stamp" in str(c).lower():
                v = pd.to_numeric(df[c], errors="coerce").to_numpy()
                d = np.diff(v[~np.isnan(v)])
                if len(d):
                    print(f"\n時刻の列 '{c}': 間隔の中央値 {np.median(d)}（単位はファイルの定義による）")
except Exception as e:
    print("\npandas では読めませんでした:", e)
