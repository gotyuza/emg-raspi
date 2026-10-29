"""
Pi 5 で「ESN と LSTM などの重いモデル」の推論速度を比べるベンチマーク。

  ・重みはランダム（速度を測るだけなので学習は不要。データセットも不要）
  ・想定：PiEEG-16 → 16ch, 1000Hz, 窓 150ms, 20ms ごとに 1 回推論（= 予算 20ms）
  ・1 回の推論にかかる時間の 平均 / 中央値 / p99（100 回に 1 回の遅い値）/ 最大 を表示
  ・結果は bench_<ホスト名>_th<スレッド数>.csv に保存

使い方（Pi 5 で）:
  source ~/emg-env/bin/activate
  pip install torch            # 最初の 1 回だけ（数分かかる）
  python bench_pi5.py                  # 標準（1 スレッド）
  python bench_pi5.py --threads 4      # 4 コア全部使った場合
  python bench_pi5.py --quick          # 回数を減らして手早く
  python bench_pi5.py --no-torch       # torch なしで numpy のものだけ

torch が無くても numpy の ESN / LSTM / 前処理だけは測れます。
"""
import argparse, os, socket, time, csv, warnings
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=1, help="推論に使う CPU スレッド数（Pi 5 は最大 4）")
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--no-torch", action="store_true")
ap.add_argument("--no-onnx", action="store_true")
a = ap.parse_args()
if a.quick:
    a.iters = 150

FS, C, WIN, HOP = 1000, 16, 150, 20      # 1kHz, 16ch, 150ms 窓, 20ms ごと
F = 2 * C                                 # 特徴量：ch ごとに log RMS と log WL → 32 次元
HIST = 50                                 # 特徴量の履歴を使うモデルは過去 1 秒（50 ステップ）を見る
BUDGET_MS = HOP * 1000 / FS               # 20 ms
WARM = 30
rng = np.random.default_rng(0)
results = []


def bench(name, kind, fn, params=0, iters=None):
    """fn() を何回も呼んで 1 回あたりの時間を測る"""
    iters = iters or a.iters
    for _ in range(WARM):
        fn()
    ts = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        ts[i] = (time.perf_counter_ns() - t0) / 1e6
    r = dict(model=name, kind=kind, params=params,
             mean_ms=ts.mean(), p50_ms=np.median(ts), p99_ms=np.percentile(ts, 99), max_ms=ts.max())
    r["budget_%"] = r["p99_ms"] / BUDGET_MS * 100
    results.append(r)
    ok = "OK " if r["p99_ms"] < BUDGET_MS * 0.5 else ("ぎり" if r["p99_ms"] < BUDGET_MS else "NG ")
    print(f"  {ok} {name:34s} {kind:9s} {params:>9,d}  平均{r['mean_ms']:7.3f}  "
          f"p99{r['p99_ms']:7.3f}  最大{r['max_ms']:7.2f} ms   予算の{r['budget_%']:5.1f}%", flush=True)


print(f"ホスト: {socket.gethostname()}   スレッド: {a.threads}   回数: {a.iters}   予算: {BUDGET_MS:.0f} ms/回")
print("  判定  モデル                             種類        パラメータ数   （OK=予算の半分未満, ぎり=予算内, NG=超過）\n")

# ---------------------------------------------------------------- 1) 前処理（どのモデルでも必要）
print("[前処理] フィルタ（20 サンプル分）＋ 150ms 窓の log RMS / log WL")
b, aa = iirnotch(50, 30, FS)
SOS = np.vstack([butter(4, [20, 450], btype="band", fs=FS, output="sos"), tf2sos(b, aa)])
zi = np.repeat(sosfilt_zi(SOS)[:, :, None], C, axis=2)
buf = np.zeros((WIN, C))
chunk = rng.standard_normal((HOP, C))


def preprocess():
    global zi, buf
    y, zi = sosfilt(SOS, chunk, axis=0, zi=zi)
    buf = np.vstack([buf[HOP:], y])
    rms = np.log(np.sqrt((buf ** 2).mean(0)) + 1e-8)
    wl = np.log(np.abs(np.diff(buf, axis=0)).sum(0) + 1e-8)
    return np.concatenate([rms, wl])


bench("filter+features", "numpy", preprocess)
u = preprocess()

# ---------------------------------------------------------------- 2) ESN（今使っているもの）
print("\n[ESN] 1 ステップ更新 + 出力（numpy）")
for N in (300, 1000, 2000):
    Win = rng.uniform(-0.5, 0.5, (N, F))
    W = rng.standard_normal((N, N)) * (rng.random((N, N)) < 0.1)
    W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W))) if N <= 1000 else 0.9 / (np.sqrt(N * 0.1))
    Wout = rng.standard_normal((5, N + 1)) * 0.01
    h = np.zeros(N)

    def esn(Win=Win, W=W, Wout=Wout):
        global h
        h = 0.7 * h + 0.3 * np.tanh(Win @ u + W @ h)
        return Wout @ np.append(h, 1.0)

    bench(f"ESN N={N}", "numpy", esn, params=Win.size + W.size + Wout.size)

# ---------------------------------------------------------------- 3) LSTM を numpy で（torch 不要）
print("\n[LSTM] 1 ステップずつ状態を持ち越す方式（numpy 手書き）")
for H, L in ((64, 1), (128, 2), (256, 2)):
    Ws = [(rng.standard_normal((4 * H, (F if l == 0 else H) + H)) * 0.1, np.zeros(4 * H)) for l in range(L)]
    Wo = rng.standard_normal((5, H)) * 0.1
    st = [(np.zeros(H), np.zeros(H)) for _ in range(L)]
    sig = lambda z: 1 / (1 + np.exp(-z))

    def lstm_np(Ws=Ws, Wo=Wo, st=st, H=H):
        x = u
        for l, (Wl, bl) in enumerate(Ws):
            hh, cc = st[l]
            z = Wl @ np.concatenate([x, hh]) + bl
            i, f, g, o = sig(z[:H]), sig(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sig(z[3 * H:])
            cc = f * cc + i * g
            hh = o * np.tanh(cc)
            st[l] = (hh, cc)
            x = hh
        return Wo @ x

    bench(f"LSTM-np H={H} L={L}", "numpy", lstm_np, params=sum(w.size + b_.size for w, b_ in Ws) + Wo.size)

# ---------------------------------------------------------------- 4) PyTorch のモデル
if not a.no_torch:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\n※ torch が入っていないので PyTorch の部分は飛ばします（pip install torch）")
        a.no_torch = True

if not a.no_torch:
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    print(f"\n[PyTorch {torch.__version__}]  ※ 'stream' = 20ms ごとに 1 ステップだけ進める / "
          f"'window' = 毎回 窓全体を計算し直す")

    class RNNStream(nn.Module):           # 状態を持ち越して 1 ステップだけ進める
        def __init__(s, cell, H, L):
            super().__init__()
            s.rnn = getattr(nn, cell)(F, H, L, batch_first=True)
            s.out = nn.Linear(H, 5)

        def forward(s, x, *state):
            y, st = s.rnn(x, state if len(state) > 1 else (state[0] if state else None))
            return (s.out(y[:, -1]),) + (st if isinstance(st, tuple) else (st,))

    class RNNWindow(nn.Module):           # 毎回 窓全体（T ステップ）を流す
        def __init__(s, cell, In, H, L):
            super().__init__()
            s.rnn = getattr(nn, cell)(In, H, L, batch_first=True)
            s.out = nn.Linear(H, 5)

        def forward(s, x):
            y, _ = s.rnn(x)
            return s.out(y[:, -1])

    class CNNRaw(nn.Module):              # 生波形 150×16 を 1D-CNN で
        def __init__(s, ch=64):
            super().__init__()
            s.net = nn.Sequential(
                nn.Conv1d(C, ch, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(ch, ch, 7, 2, 3), nn.ReLU(),
                nn.Conv1d(ch, ch, 7, 2, 3), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, 5))

        def forward(s, x):
            return s.net(x)

    class TCN(nn.Module):                 # 特徴量の履歴に膨張畳み込み
        def __init__(s, ch=64, layers=5):
            super().__init__()
            mods, cin = [], F
            for k in range(layers):
                d = 2 ** k
                mods += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(cin, ch, 3, dilation=d), nn.ReLU()]
                cin = ch
            s.net = nn.Sequential(*mods)
            s.out = nn.Linear(ch, 5)

        def forward(s, x):
            return s.out(s.net(x)[:, :, -1])

    class TFM(nn.Module):                 # 小さい Transformer
        def __init__(s, d=64, L=2):
            super().__init__()
            s.inp = nn.Linear(F, d)
            s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True), L)
            s.out = nn.Linear(d, 5)

        def forward(s, x):
            return s.out(s.enc(s.inp(x))[:, -1])

    def nparam(m):
        return sum(p.numel() for p in m.parameters())

    x1 = torch.randn(1, 1, F)
    xh = torch.randn(1, HIST, F)
    xraw = torch.randn(1, WIN, C)
    xraw_c = torch.randn(1, C, WIN)
    xh_c = torch.randn(1, F, HIST)
    onnx_jobs = []            # (名前, モデル, 入力のタプル, 状態を回すか)

    def run_stream(m, cell, H, L):
        st = [torch.zeros(L, 1, H)] + ([torch.zeros(L, 1, H)] if cell == "LSTM" else [])
        state = {"s": st}

        def f():
            with torch.inference_mode():
                o = m(x1, *state["s"])
            state["s"] = list(o[1:])
        return f

    print(" -- stream（本番で一番現実的な使い方）")
    for cell, H, L in (("LSTM", 64, 1), ("LSTM", 128, 2), ("LSTM", 256, 2), ("LSTM", 512, 3), ("GRU", 128, 2)):
        m = RNNStream(cell, H, L).eval()
        bench(f"{cell} H={H} L={L}", "stream", run_stream(m, cell, H, L), nparam(m))
        if (cell, H, L) == ("LSTM", 128, 2):
            st0 = (torch.zeros(L, 1, H), torch.zeros(L, 1, H))
            onnx_jobs.append((f"{cell} H={H} L={L}", m, (x1,) + st0, True))

    print(" -- window（毎回 窓全体を計算し直す。重い）")

    def run_win(m, x):
        def f():
            with torch.inference_mode():
                m(x)
        return f

    win_models = [
        ("LSTM feat T=50 H=128 L=2", RNNWindow("LSTM", F, 128, 2), xh),
        ("LSTM raw T=150 H=128 L=2", RNNWindow("LSTM", C, 128, 2), xraw),
        ("CNN1D raw 150x16 ch=64", CNNRaw(64), xraw_c),
        ("TCN feat T=50 ch=64", TCN(64, 5), xh_c),
        ("Transformer feat T=50 d=64", TFM(64, 2), xh),
    ]
    for name, m, x in win_models:
        m.eval()
        bench(name, "window", run_win(m, x), nparam(m))
        onnx_jobs.append((name, m, (x,), False))

    # int8 動的量子化（LSTM / Linear の重みを 8bit に）
    try:
        from torch.ao.quantization import quantize_dynamic
        print(" -- int8 動的量子化")
        for cell, H, L in (("LSTM", 256, 2), ("LSTM", 512, 3)):
            m32 = RNNStream(cell, H, L).eval()
            mq = quantize_dynamic(m32, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
            bench(f"{cell} H={H} L={L} int8", "stream", run_stream(mq, cell, H, L), nparam(m32))
    except Exception as e:
        print("  （int8 量子化はこの torch では使えませんでした:", type(e).__name__, ")")

    # ---------------------------------------------------------------- 5) ONNX Runtime
    if not a.no_onnx:
        try:
            import onnxruntime as ort
            import io
            print(f"\n[ONNX Runtime {ort.__version__}]  同じモデルを ONNX にして実行")
            so = ort.SessionOptions()
            so.intra_op_num_threads = a.threads
            so.inter_op_num_threads = 1
            for name, m, xs, stateful in onnx_jobs:
                try:
                    bio = io.BytesIO()
                    names = [f"in{i}" for i in range(len(xs))]
                    try:
                        torch.onnx.export(m, xs, bio, input_names=names, opset_version=17, dynamo=False)
                    except TypeError:
                        torch.onnx.export(m, xs, bio, input_names=names, opset_version=17)
                    sess = ort.InferenceSession(bio.getvalue(), so, providers=["CPUExecutionProvider"])
                    feed = {n: x.numpy() for n, x in zip(names, xs)}

                    if stateful:
                        def f(sess=sess, feed=feed, names=names):
                            o = sess.run(None, feed)
                            for k, n in enumerate(names[1:]):
                                feed[n] = o[1 + k]
                    else:
                        def f(sess=sess, feed=feed):
                            sess.run(None, feed)
                    bench(name, "onnx", f, nparam(m))
                except Exception as e:
                    print(f"  （{name} は ONNX 化できませんでした: {type(e).__name__}: {str(e)[:80]}）")
        except ImportError:
            print("\n※ onnxruntime が無いので ONNX は飛ばします")

# ---------------------------------------------------------------- まとめ
temp = None
try:
    temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
except Exception:
    pass
thr = None
try:
    import subprocess
    thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
except Exception:
    pass
print(f"\nCPU 温度: {temp if temp is not None else '-'} ℃   スロットリング: {thr or '-'}  "
      "（throttled=0x0 以外なら熱や電源不足で遅くなっています）")

out = f"bench_{socket.gethostname()}_th{a.threads}.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as fp:
    w = csv.DictWriter(fp, fieldnames=list(results[0].keys()))
    w.writeheader()
    for r in results:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("保存しました:", out)
