"""
ForceBand のデモデータ全体で「EMG -> 指先の力」の軽量ベースラインを評価する。
  学習: data/splits/calib.txt に載っているデモ（0-35）
  評価: data/splits/demo_val.txt に載っているデモ（36-93）
使い方（ForceBand フォルダの中で）:
  pip install h5py numpy scipy scikit-learn matplotlib
  python forceband_multi.py
"""
import glob, os, sys, time
import h5py, numpy as np
from scipy.signal import butter, sosfilt, iirnotch, tf2sos
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
FS, W, H = 1000, 150, 20            # 1kHz, 150ms 窓, 20ms ずらし
NOTCH = 60                          # 米国で記録されたデータなので 60Hz（函館で記録するなら 50）
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
USE = [0, 1]                        # つまむ動作: 親指と人差し指

SOS_BP = butter(4, [20, 450], btype="band", fs=FS, output="sos")
SOS_N = tf2sos(*iirnotch(NOTCH, 30, FS))

def find_files(split_name):
    """split ファイルの各行（名前やパス）を data/ 以下の hdf5 に対応付ける"""
    allh5 = glob.glob(os.path.join(ROOT, "data", "**", "*.hdf5"), recursive=True)
    path = os.path.join(ROOT, "data", "splits", split_name)
    names = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for n in names:
        stem = os.path.splitext(os.path.basename(n))[0]
        hit = [p for p in allh5 if os.path.splitext(os.path.basename(p))[0] == stem]
        if hit: out.append(hit[0])
    print(f"{split_name}: {len(names)} 行 -> {len(out)} ファイル見つかった")
    return sorted(set(out))

def features(path):
    with h5py.File(path, "r") as f:
        g = f["forceband/timeseries"]
        emg = g["emg"][:]
        emg = emg[:, 1:9] if emg.shape[1] > 8 else emg     # 24列で保存されている場合は 1:9 が EMG
        force = g["force"][:] / 18.0                       # センサ単位 -> N（およそ）
    x = sosfilt(SOS_N, sosfilt(SOS_BP, emg.astype(np.float64), axis=0), axis=0)  # 因果的フィルタ
    idx = np.arange(W, len(x), H)
    X = np.empty((len(idx), 16))
    for k, i in enumerate(idx):
        seg = x[i - W:i]
        X[k, :8] = np.log(np.sqrt((seg ** 2).mean(0)) + 1e-6)          # RMS
        X[k, 8:] = np.log(np.abs(np.diff(seg, axis=0)).sum(0) + 1e-6)  # 波形長 WL
    return X, force[idx]   # ラベルは窓の終端時刻（未来を使わない）

def load(files):
    XY = [features(p) for p in files]
    return np.vstack([a for a, _ in XY]), np.vstack([b for _, b in XY])

def smooth(P, a=0.3):      # 1次ローパス（リアルタイムで使える形）
    S = P.copy()
    for i in range(1, len(S)): S[i] = a * P[i] + (1 - a) * S[i - 1]
    return S

if __name__ == "__main__":
    tr, te = find_files("calib.txt"), find_files("demo_val.txt")
    if not tr or not te:
        sys.exit("hdf5 が見つかりません。ForceBand フォルダの中で実行しているか確認してください。")
    t0 = time.time()
    Xtr, Ytr = load(tr); Xte, Yte = load(te)
    print(f"学習 {len(Xtr)} 窓 / 評価 {len(Xte)} 窓  (特徴量計算 {time.time()-t0:.0f} 秒)")
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    models = {
        "Ridge": Ridge(alpha=1.0),
        "MLP 64-64": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300,
                                  early_stopping=True, random_state=0),
    }
    for name, m in models.items():
        m.fit(Xtr, Ytr[:, USE])
        P = smooth(np.clip(m.predict(Xte), 0, None))
        print(f"\n[{name}]  (評価はデモ36-93、平滑化あり)")
        for j, fi in enumerate(USE):
            y, p = Yte[:, fi], P[:, j]
            print(f"  {FINGERS[fi]:6s} R2={r2_score(y,p):5.2f}  MAE={mean_absolute_error(y,p):4.2f} N"
                  f"  r={np.corrcoef(y,p)[0,1]:4.2f}")
