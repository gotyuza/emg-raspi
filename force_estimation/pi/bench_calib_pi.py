"""
【ラズパイ側】校正（再学習）にかかる時間を測る。calib_compare.py と同じモデル・同じ学習の進め方で、
ダミーのデータ（中身は乱数、大きさは本物と同じ）を使って計算時間だけを測る。精度は測らない。

必要なもの: numpy, torch（CPU 版）。putEMG のデータは不要。

測ること（校正データの長さごとに）
  ・1 エポック（学習 → 検証）にかかる時間
  ・それを何エポック回すと何分になるか（実際のエポック数は早期終了で変わるので、10 / 20 / 40 で表示）
  ・ESN は「状態の計算 + リッジ回帰」をそのまま実行して時間を測る

使い方（ラズパイで）:
  python bench_calib_pi.py                       # 16ch、校正 1 分 / 4 分 / 11 分
  python bench_calib_pi.py --nch 24 --secs 240   # 24ch、4 分だけ
  python bench_calib_pi.py --threads 4           # 使う CPU コア数（ラズパイ 5 は 4 コア）
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--nch", type=int, default=16)
ap.add_argument("--secs", type=float, nargs="*", default=[60, 240, 684])
ap.add_argument("--epochs", type=int, nargs="*", default=[10, 20, 40])
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--models", nargs="*", default=["esn", "lstm", "cnnlstm", "cnn_fixed", "transformer"])
a = ap.parse_args()
torch.set_num_threads(a.threads)

NOUT, WIN, HOP, SEQ_T, BURN, CTX, BATCH = 10, 150, 20, 100, 10, 50, 16
nch, nin = a.nch, 2 * a.nch


# ---------------------------------------------------------------- personal_compare.py と同じモデル
class LSTMFeat(nn.Module):
    kind = "seq"

    def __init__(s, h=32, l=1):
        super().__init__()
        s.rnn = nn.LSTM(nin, h, l, batch_first=True); s.out = nn.Linear(h, NOUT)

    def forward(s, F, W=None):
        return s.out(s.rnn(F)[0])


class CNNLSTM(nn.Module):
    kind = "seq_raw"

    def __init__(s, c=64, h=128):
        super().__init__()
        s.enc = nn.Sequential(
            nn.Conv1d(nch, nch, 15, padding=7, groups=nch, bias=False), nn.BatchNorm1d(nch),
            nn.Conv1d(nch, c, 1, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.Conv1d(c, c, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.Conv1d(c, c, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(c), nn.ELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.rnn = nn.LSTM(c, h, 1, batch_first=True); s.out = nn.Linear(h, NOUT)

    def forward(s, F, W):
        B, T = W.shape[:2]
        e = s.enc(W.reshape(B * T, nch, WIN)).reshape(B, T, -1)
        return s.out(s.rnn(e)[0])


class TransformerFeat(nn.Module):
    kind = "ctx"

    def __init__(s, d=32, l=1):
        super().__init__()
        s.inp = nn.Linear(nin, d); s.pos = nn.Parameter(torch.zeros(1, CTX, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True), l)
        s.out = nn.Linear(d, NOUT)

    def forward(s, F, W=None):
        return s.out(s.enc(s.inp(F) + s.pos)[:, -1])


MODELS = {"lstm": ("LSTM S", LSTMFeat), "cnnlstm": ("CNN+LSTM L", CNNLSTM), "transformer": ("Transformer S", TransformerFeat)}


def epoch_time(model, n):
    """train_deep と同じ: 1 エポック = steps 回の学習 + 検証データ（全体の 1/5）の予測"""
    n_fit, n_val = int(n * 0.8), n - int(n * 0.8)
    bsz = BATCH * 16 if model.kind == "ctx" else BATCH
    per = 1 if model.kind == "ctx" else SEQ_T
    steps = int(np.clip(n_fit / (bsz * per) * 2, 20, 400))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def batch():
        if model.kind == "ctx":
            return torch.randn(bsz, CTX, nin), None, torch.randn(bsz, NOUT)
        W = torch.randn(bsz, SEQ_T, nch, WIN) if model.kind == "seq_raw" else None
        return torch.randn(bsz, SEQ_T, nin), W, torch.randn(bsz, SEQ_T, NOUT)

    def train_step():
        F, W, Y = batch()
        y = model(F, W)
        loss = ((y - Y) ** 2).mean() if model.kind == "ctx" else ((y - Y)[:, BURN:] ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.train()
    train_step()                                        # 1 回目は準備で遅いので除く
    k = min(steps, 5)
    t0 = time.perf_counter()
    for _ in range(k):
        train_step()
    t_train = (time.perf_counter() - t0) / k * steps
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():                               # 検証: 検証データ全体を予測
        if model.kind == "ctx":
            for i in range(0, n_val, 2048):
                model(torch.randn(min(2048, n_val - i), CTX, nin))
        elif model.kind == "seq_raw":
            e = torch.cat([model.enc(torch.randn(min(2048, n_val - i), nch, WIN)) for i in range(0, n_val, 2048)])
            model.out(model.rnn(e[None])[0])
        else:
            model(torch.randn(1, n_val, nin))
    t_val = time.perf_counter() - t0
    return t_train + t_val, steps


def cnn_fixed_time(n):
    """CNN固定の校正: CNN の出力を 1 回だけ計算しておき、LSTM と出力層だけを学習する"""
    m = CNNLSTM().eval()
    t0 = time.perf_counter()
    with torch.no_grad():                                 # 校正データ全部の窓を CNN に 1 回ずつ通す
        E = torch.cat([m.enc(torch.randn(min(1024, n - i), nch, WIN)) for i in range(0, n, 1024)])
    t_emb = time.perf_counter() - t0
    n_fit, n_val = int(n * 0.8), n - int(n * 0.8)
    steps = int(np.clip(n_fit / (BATCH * SEQ_T) * 2, 20, 400))
    params = list(m.rnn.parameters()) + list(m.out.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    m.rnn.train()

    def step():
        idx = torch.randint(0, n_fit - SEQ_T, (BATCH,))
        e = torch.stack([E[i:i + SEQ_T] for i in idx])
        y = m.out(m.rnn(e)[0])
        loss = ((y - torch.randn_like(y))[:, BURN:] ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    step()
    t0 = time.perf_counter()
    for _ in range(min(steps, 5)):
        step()
    t_train = (time.perf_counter() - t0) / min(steps, 5) * steps
    t0 = time.perf_counter()
    with torch.no_grad():
        m.out(m.rnn(E[None, n_fit:])[0])
    return t_emb, t_train + time.perf_counter() - t0, steps


def esn_time(n, N=300, lams=6):
    rng = np.random.default_rng(0)
    Win = rng.uniform(-0.3, 0.3, (N, nin))
    W = rng.normal(0, 1, (N, N)) * (rng.random((N, N)) < 10 / N)
    W *= 0.8 / np.max(np.abs(np.linalg.eigvals(W)))
    F = rng.normal(size=(n, nin)); Y = rng.normal(size=(n, NOUT))
    t0 = time.perf_counter()
    h, H = np.zeros(N), np.empty((n, N))
    for i in range(n):
        h = 0.4 * h + 0.6 * np.tanh(Win @ F[i] + W @ h); H[i] = h
    Z = np.c_[H, np.ones(n)]
    ev, Q = np.linalg.eigh(Z.T @ Z); ZtY = Z.T @ Y
    for lam in np.logspace(-2, 3, lams):
        Q @ ((Q.T @ ZtY) / (ev + lam)[:, None])
    return time.perf_counter() - t0


print(f"{nch}ch / CPU {a.threads} コア / 校正データ {', '.join(f'{s / 60:g}分' for s in a.secs)}")
print("（ダミーデータで計算時間だけを測定。実際のエポック数は早期終了で変わる）\n")
rows = []
for s in a.secs:
    n = int(s * 1000 / HOP)                                   # 20ms ごとの行数
    print(f"■ 校正データ {s / 60:g} 分（{n} 行）")
    if "esn" in a.models:
        t = esn_time(n)
        print(f"  {'ESN M':14s} 状態の計算＋リッジ回帰: {t:6.1f} 秒（エポックなし）")
    if "cnn_fixed" in a.models:
        te, t, steps = cnn_fixed_time(n)
        est = "  ".join(f"{e}エポック {(te + t * e) / 60:5.1f}分" for e in a.epochs)
        print(f"  {'CNN+LSTM CNN固定':14s} CNN の出力計算 {te:5.1f} 秒（最初に 1 回）＋ 1 エポック {t:5.2f} 秒 → {est}", flush=True)
    for key in [m for m in a.models if m in MODELS]:
        name, cls = MODELS[key]
        torch.manual_seed(0)
        t, steps = epoch_time(cls(), n)
        est = "  ".join(f"{e}エポック {t * e / 60:5.1f}分" for e in a.epochs)
        print(f"  {name:14s} 1 エポック {t:6.1f} 秒（学習 {steps} 回＋検証） → {est}", flush=True)
    print()
try:
    print(f"CPU 温度: {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.1f} ℃")
except Exception:
    pass
