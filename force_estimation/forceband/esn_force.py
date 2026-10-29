"""
ESN（Echo State Network）で、筋電から親指・人差し指の力を推定する。

  学習: data/splits/calib.txt のデモ（0-35）
  評価: data/splits/demo_val.txt のデモ（36-93）

使い方（ForceBand フォルダの中で）:
  pip install h5py numpy scipy matplotlib
  python esn_force.py                       # 学習 → 校正 → 評価 → esn_model.npz に保存
  python esn_force.py --plot                # グラフも表示
  python esn_force.py --sweep               # Ridge の正則化を何通りか試して、校正用データで一番良いものを使う
  python esn_force.py --target fingers      # 親指と人差し指を別々に推定（初期設定は「つまむ強さ」1 つ）
  python esn_force.py --rho 0.95 --leak 0.5 # リザーバーのハイパーパラメータを変える
  python esn_force.py --imu gate            # IMU を接触判定にだけ使う（all でリザーバーにも入れる）
  python esn_force.py --freq                # 周波数の特徴量も追加

2 段構えの推定（同じリザーバーに読み出しを 2 つ付ける）:
  ① 接触判定 : 「つまんでいるか」を 0〜1 で出す（実測 > 0.5N を 1 として学習）
  ② 力の推定 : つまんでいる区間のデータだけで学習（0N の区間に引っ張られて小さく出るのを防ぐ）
  出力 = ① が ON のときだけ ②、それ以外は 0
  ① の ON/OFF はヒステリシス・最短時間・なめらかな切り替えで、バタつきと急な途切れを防ぐ
  しきい値は、学習用デモの最後の数本（校正用）で自動で決める。評価用デモは最後まで使わない。

処理の流れ:
  EMG(250Hz)（＋IMU）→ 因果フィルタ → 150ms窓の特徴量(20msごと) → 標準化 → リザーバー → Ridge読み出し → 後処理 → 力
  すべて「未来のデータを使わない」処理なので、そのままリアルタイムに移せる。
"""
import argparse, glob, os, sys, time
import h5py
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos

# ---------------------------------------------------------------- 設定
UPSAMPLE = 4          # ForceBand は 250Hz の値を 4 回ずつ繰り返して 1kHz にしている
FS = 1000 // UPSAMPLE # 元のサンプリングレート 250Hz
WIN = 38              # 窓 150ms ≒ 38 サンプル
HOP = 5               # 20ms ごとに 1 回推定
NOTCH = 60            # 米国で記録されたデータ。函館で記録するなら 50
FINGERS = [0, 1]      # 0:親指 1:人差し指
NAMES = ["thumb", "index", "middle", "ring", "pinky"]
WASHOUT = 25          # 各ファイルの最初 25 ステップ(0.5 秒)は状態が落ち着かないので学習に使わない

SOS = np.vstack([butter(4, [20, 120], btype="band", fs=FS, output="sos"),
                 tf2sos(*iirnotch(NOTCH, 30, FS))])


# ---------------------------------------------------------------- 前処理
# 特徴量の設定（main で引数から書き換える）
OPT = dict(imu="none", freq=False)   # imu: "none" / "all"(リザーバーに入れる) / "gate"(接触判定にだけ使う)


def load(path):
    """hdf5 を読み、250Hz に戻した EMG(8ch)・IMU(10)・力(N) を返す"""
    with h5py.File(path, "r") as f:
        g = f["forceband/timeseries"]
        emg = g["emg"][:, 1:9][::UPSAMPLE].astype(np.float64)
        imu = g["imu"][:][::UPSAMPLE].astype(np.float64)
        force = (g["force"][:] / 18.0)[::UPSAMPLE]
    return emg, imu, force


def n_features():
    return 16 + (8 if OPT["imu"] != "none" else 0) + (16 if OPT["freq"] else 0)


def features(emg, imu):
    """窓ごとの特徴量 → (ステップ数, n_features())
      EMG  : log RMS(8) と log 波形長 WL(8)                                  … 筋活動の大きさ
      IMU  : ジャイロの各軸の平均(3)・回転の速さの平均(1)                     … 腕を動かしているか
             加速度の各軸の平均(3)＝重力の向き＝前腕の姿勢・加速度の大きさのばらつき(1) … 姿勢と揺れ
             （クォータニオンは向きの基準が記録ごとに違いうるので使わない）
      周波数（--freq）: ゼロ交差率 ZC(8)・傾き符号変化率 SSC(8)            … 周波数の高さの目安"""
    x = sosfilt(SOS, emg, axis=0)
    idx = np.arange(WIN, len(x) + 1, HOP)
    F = np.empty((len(idx), n_features()))
    gyro, acc = imu[:, 4:7], imu[:, 7:10]
    for k, i in enumerate(idx):
        seg = x[i - WIN:i]
        rms = np.sqrt((seg ** 2).mean(0))
        F[k, :8] = np.log(rms + 1e-6)
        F[k, 8:16] = np.log(np.abs(np.diff(seg, axis=0)).sum(0) + 1e-6)
        c = 16
        if OPT["imu"] != "none":
            gw, aw = gyro[i - WIN:i], acc[i - WIN:i]
            F[k, c:c + 3] = gw.mean(0)
            F[k, c + 3] = np.linalg.norm(gw, axis=1).mean()
            F[k, c + 4:c + 7] = aw.mean(0)
            F[k, c + 7] = np.linalg.norm(aw, axis=1).std()
            c += 8
        if OPT["freq"]:
            th = 0.05 * rms                                   # 小さな揺れは数えない
            d = np.diff(seg, axis=0)
            zc = ((seg[:-1] * seg[1:] < 0) & (np.abs(seg[:-1] - seg[1:]) > th)).mean(0)
            ssc = ((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) > th) | (np.abs(d[1:]) > th))).mean(0)
            F[k, c:c + 8], F[k, c + 8:c + 16] = zc, ssc
    return F, idx - 1          # idx-1: 各特徴量の「窓の最後のサンプル」の位置（ラベルの時刻）


# ---------------------------------------------------------------- ESN 本体
class ESN:
    """
    状態更新:  h(t) = (1-a) h(t-1) + a tanh( Win u(t) + W h(t-1) )
      rho      : W のスペクトル半径（記憶の長さ。1 に近いほど長い）
      leak (a) : リーク率（小さいほど状態がゆっくり変わる = 記憶が長い）
      in_scale : 入力の強さ（大きいほど tanh が飽和して非線形になる）
    学習するのは読み出しだけ（Ridge 回帰、閉じた式で一瞬）。このスクリプトでは TwoStage が読み出しを持つ。
    """

    def __init__(self, n_in, n_res=300, rho=0.9, leak=1.0, in_scale=0.3,
                 density=0.1, ridge=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.Win = rng.uniform(-1, 1, (n_res, n_in)) * in_scale
        W = rng.uniform(-1, 1, (n_res, n_res)) * (rng.random((n_res, n_res)) < density)
        W *= rho / np.max(np.abs(np.linalg.eigvals(W)))     # スペクトル半径を rho に合わせる
        self.W, self.leak, self.ridge = W, leak, ridge
        self.h = np.zeros(n_res)
        self.Wout = None

    def reset(self):
        self.h[:] = 0

    def step(self, u):
        """1 ステップ進めて状態を返す（リアルタイム用）"""
        pre = self.Win @ u + self.W @ self.h
        self.h = (1 - self.leak) * self.h + self.leak * np.tanh(pre)
        return self.h

    def run(self, U):
        """系列をまとめて流して、全時刻の状態を返す"""
        self.reset()
        return np.array([self.step(u).copy() for u in U])

    def fit(self, H, Y):
        """Ridge 回帰: Wout = Y^T Z (Z^T Z + λI)^-1   （Z は状態にバイアス 1 を足したもの）"""
        Z = np.hstack([H, np.ones((len(H), 1))])
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.Wout = np.linalg.solve(A, Z.T @ Y).T

    def readout(self, h):
        return self.Wout[:, :-1] @ h.T + self.Wout[:, -1:] if h.ndim > 1 else \
               self.Wout[:, :-1] @ h + self.Wout[:, -1]



# ---------------------------------------------------------------- データ準備
def split_files(root, name):
    allh5 = glob.glob(os.path.join(root, "data", "**", "*.hdf5"), recursive=True)
    stems = {os.path.splitext(os.path.basename(p))[0]: p for p in allh5}
    lines = [l.strip() for l in open(os.path.join(root, "data", "splits", name), encoding="utf-8") if l.strip()]
    files = [stems[s] for s in (os.path.splitext(os.path.basename(l))[0] for l in lines) if s in stems]
    print(f"{name}: {len(files)} ファイル")
    return files


def target_of(force, target):
    """推定する対象を選ぶ。
    pinch  : 親指の力を「つまむ強さ」として 1 つだけ推定（親指は他の指すべてと向かい合うので全体の強さを表しやすい）
    fingers: 親指と人差し指を別々に推定"""
    return force[:, [0]] if target == "pinch" else force[:, FINGERS]


def prepare(files, target):
    """ファイルごとに (特徴量, 力ラベル) のリストを返す。ファイルをまたいで状態はつなげない"""
    out = []
    for p in files:
        emg, imu, force = load(p)
        F, last = features(emg, imu)
        out.append((F, target_of(force[last], target)))
    return out


IMU_COLS = slice(16, 24)   # 特徴量の中の IMU の列


def reservoir_cols():
    """リザーバーに入れる特徴量の列（--imu gate のときは IMU を除く）"""
    cols = np.arange(n_features())
    return np.delete(cols, np.arange(16, 24)) if OPT["imu"] == "gate" else cols


def run_states(esn, data, mu, sd):
    """ファイルごとの (読み出しへの入力, ラベル)。最初の WASHOUT ステップは捨てる。
    読み出しへの入力 = [リザーバーの状態 | (--imu gate のとき) 標準化した IMU 特徴量]"""
    out = []
    for F, y in data:
        Fs = (F - mu) / sd
        H = esn.run(Fs[:, reservoir_cols()])
        if OPT["imu"] == "gate":
            H = np.hstack([H, Fs[:, IMU_COLS]])
        out.append((H[WASHOUT:], y[WASHOUT:]))
    return out


def stack(pairs):
    return np.vstack([h for h, _ in pairs]), np.vstack([y for _, y in pairs])


# ---------------------------------------------------------------- 後処理と 2 段構えの推定
def smooth(P, a):
    """1 次ローパス（因果的）。a が小さいほど滑らかだが遅れる。a=1 で何もしない"""
    S = np.array(P, dtype=float)
    for i in range(1, len(S)):
        S[i] = a * S[i] + (1 - a) * S[i - 1]
    return S


def ridge_fit(H, Y, lam):
    """Ridge 回帰（バイアス付き）の閉じた式。戻り値は (状態数+1, 出力数)"""
    Z = np.hstack([H, np.ones((len(H), 1))])
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)


def apply(Wo, H):
    return np.hstack([H, np.ones((len(H), 1))]) @ Wo if H.ndim > 1 else np.append(H, 1.0) @ Wo


class TwoStage:
    """同じリザーバー状態から、接触判定と力の推定の 2 つを読み出す。

    接触判定のゲートは義手で安全に使えるように次の工夫をしている（すべて未来を使わない処理）:
      ・ヒステリシス : ON にするしきい値 thr_on と OFF にするしきい値 thr_off(< thr_on) を分ける
                       → しきい値付近で ON/OFF がバタつかない
      ・最短時間     : 一度 ON になったら MIN_ON ステップ、OFF になったら MIN_OFF ステップは切り替えない
      ・なめらかな切り替え : 出力の倍率 g を 0↔1 の間で RAMP ずつ変える（急に 0 に落とさない＝急に物を放さない）
    """
    GATE_SMOOTH = 0.15   # 接触判定の平滑化（力の推定より強めにかける）
    MIN_ON = 10          # 200ms
    MIN_OFF = 5          # 100ms
    RAMP = 0.2           # 1 ステップで倍率を 0.2 ずつ変える → 100ms で切り替わる

    def __init__(self, lam, nr):
        self.lam, self.nr, self.thr_on, self.thr_off = lam, nr, 0.6, 0.4   # nr: リザーバーのニューロン数

    # 接触判定は入力全体（IMU を含みうる）、力の推定はリザーバーの状態だけを使う

    def fit(self, H, Y):
        C = (Y.max(1, keepdims=True) > 0.5).astype(float)       # つまんでいれば 1
        self.Wc = ridge_fit(H, C, self.lam)
        on = C[:, 0] > 0
        R = H[:, :self.nr]
        self.Wf = ridge_fit(R[on], Y[on], self.lam)              # つまんでいる区間だけで学習
        self.Wb = ridge_fit(R, Y, self.lam)                      # 比較用：従来の 1 段

    def gate(self, c):
        """接触判定の系列 c から、0〜1 の倍率の系列を作る（ヒステリシス＋最短時間＋ランプ）"""
        g = np.zeros(len(c)); state, count, level = False, 0, 0.0
        for t, v in enumerate(c):
            count += 1
            if not state and v > self.thr_on and count >= self.MIN_OFF:
                state, count = True, 0
            elif state and v < self.thr_off and count >= self.MIN_ON:
                state, count = False, 0
            level = min(1.0, level + self.RAMP) if state else max(0.0, level - self.RAMP)
            g[t] = level
        return g

    def predict_file(self, H, a, two_stage=True):
        R = H[:, :self.nr]
        if not two_stage:
            return np.clip(smooth(apply(self.Wb, R), a), 0, None)
        c = smooth(apply(self.Wc, H), self.GATE_SMOOTH)[:, 0]
        f = np.clip(smooth(apply(self.Wf, R), a), 0, None)
        return f * self.gate(c)[:, None]

    def predict(self, pairs, a, two_stage=True):
        P = np.vstack([self.predict_file(h, a, two_stage) for h, _ in pairs])
        return P, np.vstack([y for _, y in pairs])

    def tune_threshold(self, cal, a):
        """校正用データでしきい値を選ぶ。
        目的 = MAE + 2 × 途切れ率（つまんでいる最中に出力が落ちる割合。義手では物を落とすことになるので罰を重くする）"""
        best = None
        for on in np.arange(0.3, 0.81, 0.05):
            for gap in [0.0, 0.1, 0.2, 0.3]:
                self.thr_on, self.thr_off = on, on - gap
                P, Y = self.predict(cal, a)
                score = np.abs(Y - P).mean() + 2 * dropout(Y[:, 0], P[:, 0])
                if best is None or score < best[0]:
                    best = (score, on, on - gap)
        _, self.thr_on, self.thr_off = best


# ---------------------------------------------------------------- 評価
def lag_ms(y, p, maxl=40):
    """推定が正解に対して何 ms 遅れているか（相互相関のピーク）"""
    y, p = y - y.mean(), p - p.mean()
    c = [np.dot(y[:len(y) - l], p[l:]) for l in range(maxl)]
    return int(np.argmax(c)) * HOP * 1000 // FS


def dropout(y, p):
    """つまんでいる最中(実測 > 2N)なのに、出力が 0.5N 未満に落ちている時間の割合"""
    m = y > 2.0
    return (p[m] < 0.5).mean() if m.any() else 0.0


def false_starts(y, p):
    """触っていない区間(実測 < 0.3N)で、出力が 0.5N を超えて立ち上がった回数（1 分あたり）"""
    rest = y < 0.3
    rise = (p[1:] >= 0.5) & (p[:-1] < 0.5) & rest[1:]
    return rise.sum() / (len(y) * HOP / FS / 60)


def metrics(y, p):
    rest = y < 0.3                    # 触っていない区間
    strong = y > 5.0                  # 強くつまんでいる区間
    return dict(
        r=np.corrcoef(y, p)[0, 1] if p.std() > 0 else 0.0,
        mae=np.abs(y - p).mean(),
        lag=lag_ms(y, p),
        rest=p[rest].mean() if rest.any() else np.nan,                       # 触っていないのに出る力 [N]
        ratio=p[strong].mean() / y[strong].mean() if strong.any() else np.nan,  # 強い力をどれだけ出せているか（1 が理想）
        drop=dropout(y, p),
        fs=false_starts(y, p),
    )


def report(title, Y, P, names):
    print(f"\n[{title}]")
    print("          r    MAE[N] 遅れ[ms] 非接触時の誤出力[N] 強い力の再現率 途切れ率 誤った立ち上がり[回/分]")
    for j, n in enumerate(names):
        m = metrics(Y[:, j], P[:, j])
        print(f"  {n:6s} {m['r']:5.2f} {m['mae']:6.2f} {m['lag']:7d} {m['rest']:13.2f} {m['ratio']:14.2f}"
              f" {m['drop']:8.2f} {m['fs']:14.1f}")


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--target", choices=["pinch", "fingers"], default="pinch")
    ap.add_argument("--n_res", type=int, default=300)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--leak", type=float, default=1.0)
    ap.add_argument("--in_scale", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true", help="Ridge の正則化を校正用データで自動で選ぶ")
    ap.add_argument("--smooth", type=float, default=0.5, help="平滑化の強さ（1 で平滑化なし）")
    ap.add_argument("--gate_smooth", type=float, default=0.15, help="接触判定の平滑化（大きいほど反応が速いがバタつきやすい）")
    ap.add_argument("--imu", choices=["none", "all", "gate"], default="none",
                    help="IMU の使い方: none=使わない / all=リザーバーに入れる / gate=接触判定にだけ使う")
    ap.add_argument("--freq", action="store_true", help="周波数の特徴量（ZC・SSC）を追加する")
    ap.add_argument("--hold", type=int, default=6, help="学習用デモの最後の何本を校正用に取り分けるか")
    ap.add_argument("--plot", action="store_true", help="評価データの一部をグラフ表示")
    a = ap.parse_args()
    OPT["imu"], OPT["freq"] = a.imu, a.freq
    names = ["pinch"] if a.target == "pinch" else [NAMES[i] for i in FINGERS]

    tr_files = split_files(a.root, "calib.txt")
    te_files = split_files(a.root, "demo_val.txt")
    if not tr_files or not te_files:
        sys.exit("hdf5 が見つかりません。ForceBand フォルダで実行してください。")
    if len(tr_files) <= a.hold:
        sys.exit("学習用のファイルが少なすぎます。--hold を小さくしてください。")

    # 学習用デモを「読み出しの学習用」と「しきい値などの校正用」に分ける（評価用デモは最後まで使わない）
    fit_files, cal_files = tr_files[:-a.hold], tr_files[-a.hold:]
    print(f"読み出しの学習 {len(fit_files)} 本 / 校正 {len(cal_files)} 本 / 評価 {len(te_files)} 本")

    t0 = time.time()
    fit, cal, te = (prepare(f, a.target) for f in (fit_files, cal_files, te_files))
    allF = np.vstack([F for F, _ in fit])
    mu, sd = allF.mean(0), allF.std(0) + 1e-8          # 標準化は学習データの統計だけで決める
    print(f"特徴量の計算 {time.time() - t0:.1f} 秒 / 特徴量 {n_features()} 次元"
          f"（EMG 16{' + IMU 8' if OPT['imu'] != 'none' else ''}{'（接触判定のみ）' if OPT['imu'] == 'gate' else ''}{' + 周波数 16' if OPT['freq'] else ''}）")

    esn = ESN(len(reservoir_cols()), a.n_res, a.rho, a.leak, a.in_scale, ridge=a.ridge)
    fit_s, cal_s, te_s = (run_states(esn, d, mu, sd) for d in (fit, cal, te))
    Hfit, Yfit = stack(fit_s)

    TwoStage.GATE_SMOOTH = a.gate_smooth

    if a.sweep:
        print("\n[Ridge の正則化の比較（校正用データ、2 段構え）]")
        results = []
        for lam in [1e-2, 1e-1, 1, 10, 100, 1000]:
            m = TwoStage(lam, a.n_res); m.fit(Hfit, Yfit); m.tune_threshold(cal_s, a.smooth)
            P, Y = m.predict(cal_s, a.smooth)
            results.append((np.abs(Y - P).mean(), lam))
            print(f"  ridge={lam:<6g} MAE={results[-1][0]:.3f} N")
        a.ridge = min(results)[1]
        print(f"  → ridge={a.ridge:g} を使う")

    model = TwoStage(a.ridge, a.n_res)
    model.fit(Hfit, Yfit)
    model.tune_threshold(cal_s, a.smooth)

    tau = 1000 / FS * HOP / (a.leak * max(1 - a.rho, 1e-3))
    print(f"\nrho={a.rho} leak={a.leak} in_scale={a.in_scale} ridge={a.ridge:g} smooth={a.smooth} gate_smooth={a.gate_smooth}"
          f"  (記憶の目安 ≒ {tau:.0f} ms) / 接触判定 ON {model.thr_on:.2f}・OFF {model.thr_off:.2f}")

    P1, Yte = model.predict(te_s, a.smooth, two_stage=False)
    P2, _ = model.predict(te_s, a.smooth)
    report("評価データ：従来の 1 段（平滑化のみ）", Yte, P1, names)
    report("評価データ：2 段構え（接触判定 → 力の推定）", Yte, P2, names)

    # 1 ステップあたりの計算時間（リアルタイムで使えるかの目安）
    u = np.zeros(len(reservoir_cols())); extra = np.zeros(8 if OPT["imu"] == "gate" else 0)
    t0 = time.perf_counter()
    for _ in range(1000):
        h = esn.step(u); apply(model.Wc, np.concatenate([h, extra])); apply(model.Wf, h)
    print(f"\n1 ステップの推論 {(time.perf_counter() - t0):.3f} ms（更新間隔 {HOP * 1000 // FS} ms）")

    np.savez("esn_model.npz", Win=esn.Win, W=esn.W, leak=esn.leak, Wc=model.Wc, Wf=model.Wf,
             thr_on=model.thr_on, thr_off=model.thr_off, smooth=a.smooth, mu=mu, sd=sd, fs=FS, win=WIN, hop=HOP,
             notch=NOTCH, target=a.target, imu=OPT['imu'], freq=OPT['freq'])
    print("esn_model.npz に保存しました")

    if a.plot:
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
        n = min(3000, len(Yte)); tt = np.arange(n) * HOP / FS
        fig, ax = plt.subplots(len(names), 1, figsize=(12, 3 + 2 * len(names)), sharex=True, squeeze=False)
        for j, nm in enumerate(names):
            ax[j, 0].plot(tt, Yte[:n, j], "k", lw=1.2, label="実測")
            ax[j, 0].plot(tt, P1[:n, j], color="#9bbcd8", lw=0.8, label="ESN 1 段")
            ax[j, 0].plot(tt, P2[:n, j], color="#d9622b", lw=1.1, label="ESN 2 段構え")
            ax[j, 0].set_ylabel(f"{nm} [N]")
        ax[0, 0].legend(loc="upper right"); ax[-1, 0].set_xlabel("time [s]（評価デモを連結）")
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
