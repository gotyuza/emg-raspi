"""
【ラズパイ側】録画した筋電を 20ms ごとに流し込み、本番と同じ流れで推定して「精度」と「時間」を測る。
必要なもの: numpy, scipy, onnxruntime（torch は不要）

本番と同じ流れ（20ms ごと）:
  筋電 20 サンプル受け取り → フィルタ（状態を持ち越す）→ 直近 150ms の log RMS / log WL
  → 標準化 → 推論（ONNX Runtime / numpy）→ 力 10ch

使い方（ラズパイで。export フォルダを ~/ に送ってある前提）:
  python pi_replay.py export/model_transformer_S_03.npz              # できるだけ速く流す（精度と処理時間）
  python pi_replay.py export/model_transformer_S_03.npz --realtime   # 本当に 20ms ごとに流す（締め切りを守れるか）
  python pi_replay.py export/model_esn_M_03.npz
  python pi_replay.py export/model_cnnlstm_L_03.npz
  python pi_replay.py export/model_ridge_03.npz
  python pi_replay.py export/model_transformer_S_03.npz --realtime --udp 10.222.251.14:5005
                                         # PC に結果を送り、live_view.py でリアルタイム表示（IP は PC のもの）

表示するもの
  ・精度: NRMSE / 相関 r（PC で計算した値と並べる）
  ・PC の予測値との差（同じ答えが出ているか）
  ・1 回あたりの処理時間: 前処理 / 推論 / 合計（平均・p99・最大）と、20ms の予算に対する割合
  ・--realtime のとき: 締め切り（次のデータが来るまで）に間に合わなかった回数
"""
import argparse, os, time
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

ap = argparse.ArgumentParser()
ap.add_argument("model", help="export_model.py が書き出した model_*.npz")
ap.add_argument("--replay", help="replay_*.npz（省略すると同じフォルダから被験者番号で探す）")
ap.add_argument("--realtime", action="store_true", help="20ms ごとに待ちながら流す")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--udp", help="結果を送る先 PCのIP:ポート（例 10.222.251.14:5005）。live_view.py で表示する")
a = ap.parse_args()
sock = None
if a.udp:
    import socket, struct
    host, port = a.udp.rsplit(":", 1)
    sock, dest = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))

B = dict(np.load(a.model, allow_pickle=True))
kind = str(B["kind"])
subj = os.path.basename(a.model).rsplit("_", 1)[-1].split(".")[0]
rp = a.replay or os.path.join(os.path.dirname(a.model), f"replay_{subj}.npz")
R = np.load(rp)
emg, Y, mask = R["emg"].astype(np.float64), R["Y"], R["mask"]
pred_pc = B["pred_pc"]                                     # PC で同じモデルが出した予測値（比べる用）
SOS, WIN, HOP, CTX = B["sos"], int(B["win"]), int(B["hop"]), int(B["ctx"])
mu, sd, scale = B["mu"], B["sd"], B["scale"]
nch = emg.shape[1]
print(f"モデル {os.path.basename(a.model)}（{kind}）/ 再生データ {os.path.basename(rp)}: "
      f"{len(emg) / 1000:.0f} 秒・{nch}ch / {'リアルタイム（20ms ごと）' if a.realtime else 'できるだけ速く'}")

# ---------------------------------------------------------------- 推論の準備
DEEP = ("transformer", "lstm", "cnnlstm")                 # 出力を scale 倍して %MVC に戻すモデル
if kind in DEEP:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(a.model.replace(".npz", ".onnx"), so, providers=["CPUExecutionProvider"])
if kind == "transformer":
    ctx = np.zeros((1, CTX, len(mu)), np.float32)            # 過去 1 秒の特徴量（足りない分は 0 = 平均）

    def infer(f):
        ctx[0, :-1] = ctx[0, 1:]; ctx[0, -1] = f
        return sess.run(None, {"x": ctx})[0][0]
elif kind == "lstm":
    st = {"h": np.zeros((int(B["layers"]), 1, int(B["hidden"])), np.float32)}
    st["c"] = st["h"].copy()

    def infer(f):
        y, st["h"], st["c"] = sess.run(None, {"x": f[None, None, :], "h": st["h"], "c": st["c"]})
        return y[0]
elif kind == "cnnlstm":                                  # 生の筋電の窓（フィルタ後 / 標準偏差）を入力にする
    rsd = B["rsd"]
    st = {"h": np.zeros((int(B["layers"]), 1, int(B["hidden"])), np.float32)}
    st["c"] = st["h"].copy()

    def infer(f):
        w = (buf / rsd).T[None].astype(np.float32)          # (1, ch, 150)
        y, st["h"], st["c"] = sess.run(None, {"w": w, "h": st["h"], "c": st["c"]})
        return y[0]
elif kind == "esn":
    Win, W, leak, Wo = B["Win"], B["W"], float(B["leak"]), B["Wo"]
    h = np.zeros(W.shape[0])

    def infer(f):
        global h
        h = (1 - leak) * h + leak * np.tanh(Win @ f + W @ h)
        return np.append(h, 1.0) @ Wo
else:   # ridge
    Wo = B["Wo"]

    def infer(f):
        return np.append(f, 1.0) @ Wo

# ---------------------------------------------------------------- 20ms ごとのループ
zi = np.repeat(sosfilt_zi(SOS)[:, :, None], nch, axis=2) * 0.0    # PC と同じく 0 から始める
buf = np.zeros((WIN, nch))
outs, t_pre, t_inf, misses = [], [], [], 0
# 最初は 150ms ぶんをまとめて受け取り、以後 20 サンプル（20ms）ずつ
starts = [0] + list(range(WIN, len(emg) - HOP + 1, HOP))
next_tick = time.perf_counter()
for s in starts:
    chunk = emg[s:s + (WIN if s == 0 else HOP)]
    if a.realtime:                                         # 次のデータが「届く」時刻まで待つ
        while time.perf_counter() < next_tick:
            pass
    t0 = time.perf_counter()
    y, zi = sosfilt(SOS, chunk, axis=0, zi=zi)
    buf = np.vstack([buf, y])[-WIN:]
    rms = np.log(np.sqrt((buf ** 2).mean(0)) + 1e-6)
    wl = np.log(np.abs(np.diff(buf, axis=0)).sum(0) + 1e-6)
    f = ((np.concatenate([rms, wl]) - mu) / sd).astype(np.float32)
    t1 = time.perf_counter()
    out = infer(f)
    t2 = time.perf_counter()
    outs.append(np.asarray(out, np.float64))
    if sock is not None and len(outs) <= len(Y):               # PC へ：時刻・電極ごとの筋電の大きさ・正解 10ch・予測 10ch
        i = len(outs) - 1
        p_now = outs[-1] * (scale if kind in DEEP else 1.0)
        try:
            sock.sendto(struct.pack(f"<IfH{nch + 20}f", i, (WIN + i * HOP) / 1000, nch,
                                    *np.exp(rms), *Y[i], *p_now), dest)
        except OSError:
            pass
    t_pre.append((t1 - t0) * 1000); t_inf.append((t2 - t1) * 1000)
    if a.realtime:
        if t2 > next_tick + HOP / 1000:
            misses += 1
        next_tick += HOP / 1000

P = np.array(outs)
if kind in DEEP:
    P = P * scale                                          # 学習時に力を幅で割っていたので戻す
n = min(len(P), len(Y)); P, Yt, Pc, k = P[:n], Y[:n], pred_pc[:n], mask[:n]

# ---------------------------------------------------------------- 結果
def scores(Yv, Pv):
    nr = np.mean(np.sqrt(((Yv - Pv) ** 2).mean(0)) / scale)
    rs = [np.corrcoef(Yv[:, i], Pv[:, i])[0, 1] for i in range(Yv.shape[1]) if Yv[:, i].std() > 0 and Pv[:, i].std() > 0]
    return nr, np.mean(rs)

nr_pi, r_pi = scores(Yt[k], P[k]); nr_pc, r_pc = scores(Yt[k], Pc[k])
diff = np.abs(P[k] - Pc[k]).max()
print(f"\n[精度]（テスト {k.sum()} ステップ）")
print(f"  ラズパイ: NRMSE {nr_pi:.3f}  r {r_pi:.3f}")
print(f"  PC      : NRMSE {nr_pc:.3f}  r {r_pc:.3f}")
print(f"  PC の予測値との差: 最大 {diff:.4f} %MVC  "
      + ("（同じ答えが出ている）" if diff < 0.5 else "（差が大きい。前処理の違いなどを確認）"))

tot = np.array(t_pre) + np.array(t_inf)
print(f"\n[1 回あたりの処理時間]（{len(tot)} 回、予算 {HOP} ms）")
for name, t in [("前処理（フィルタ＋特徴量）", np.array(t_pre)), ("推論", np.array(t_inf)), ("合計", tot)]:
    print(f"  {name:16s} 平均 {t.mean():6.3f}  p99 {np.percentile(t, 99):6.3f}  最大 {t.max():6.2f} ms"
          f"   予算の {np.percentile(t, 99) / HOP * 100:5.1f}%（p99）")
if a.realtime:
    print(f"\n[リアルタイム] 締め切りに間に合わなかった回数: {misses} / {len(tot)}")
try:
    temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    print(f"\nCPU 温度: {temp:.1f} ℃")
except Exception:
    pass
