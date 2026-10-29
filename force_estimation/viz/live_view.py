"""
【PC 側】ラズパイ（pi_replay.py --udp）から送られてくる結果を、リアルタイムでグラフに表示する。

表示（直近 10 秒が右から左へ流れる）
  上段左 : バンドごと（肘側・中央・手首側、各 8 電極）の筋電の大きさの推移
  上段右 : 今この瞬間の、電極ごとの筋電の大きさ（棒グラフ、バンドごとに色分け）
  中段 : 力センサー全体の平均 [%MVC]   … 黒 = 正解、青 = 予測
  下段 : 選んだセンサー 2 つ [%MVC]    … 実線 = 正解、点線 = 予測
  右上 : 直近 10 秒の相関 r と、ラズパイから届いた件数

使い方
  1. PC で先に起動:   python live_view.py
     （初回に Windows のファイアウォールの確認が出たら「アクセスを許可」）
  2. PC の IP を確認: ipconfig（「IPv4 アドレス」）
  3. ラズパイで:      python pi_replay.py export/model_transformer_S_03.npz --realtime --udp <PCのIP>:5005

オプション: --port 5005 / --window 10（秒）/ --chs 1 5（下段に出すセンサー番号）
"""
import argparse, socket, struct, collections, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5005)
ap.add_argument("--window", type=float, default=10.0)
ap.add_argument("--chs", type=int, nargs=2, help="下段に出すセンサー番号（1〜10）。省略すると最初の 5 秒で力の変化が大きい 2 つを選ぶ")
ap.add_argument("--map", help="finger_map.py が作った finger_map_XX.json。付けると下段に 親指 と 人差し指 を表示")
ap.add_argument("--save_png", help="（動作確認用）表示せず、指定秒数だけ受け取って画像に保存: 秒数,ファイル名")
a = ap.parse_args()

plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
HEAD = struct.Struct("<IfH")                                 # 番号・時刻・電極の数 → 続けて 電極ごとの大きさ・正解 10・予測 10
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", a.port))
sock.setblocking(False)
N = int(a.window * 50) + 10
buf = {k: collections.deque(maxlen=N) for k in ["t", "rms", "y", "p"]}
count = [0]


def receive():
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except (BlockingIOError, OSError):
            return
        if len(data) < HEAD.size:
            continue
        i, t, nch = HEAD.unpack_from(data)
        if len(data) != HEAD.size + 4 * (nch + 20):
            continue
        v = np.frombuffer(data, "<f4", offset=HEAD.size)
        buf["t"].append(t); buf["rms"].append(v[:nch])
        buf["y"].append(v[nch:nch + 10]); buf["p"].append(v[nch + 10:])
        count[0] += 1


fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.2], height_ratios=[1.3, 1.4, 1.4])
ax = [fig.add_subplot(gs[0, 0])]
ax += [fig.add_subplot(gs[1, :], sharex=ax[0]), fig.add_subplot(gs[2, :], sharex=ax[0])]
axb = fig.add_subplot(gs[0, 1])
BCOL = ["#d9622b", "#8a4bd9", "#2b7bd9"]
img = {"lines": None, "bars": None, "ymax": 1e-9}
l_ym, = ax[1].plot([], [], color="black", lw=1.6, label="正解（力センサー）")
l_pm, = ax[1].plot([], [], color="#2b7bd9", lw=1.6, label="予測（ラズパイ）")
FING = None
if a.map:                                                       # 指ごとのセンサー番号（1〜10）
    import json
    m = json.load(open(a.map, encoding="utf-8"))["sensor_to_finger"]
    FING = [[int(k) - 1 for k, v in m.items() if v == f] for f in ["thumb", "index"]]
sel = {"c": list(a.chs) if a.chs else [1, 2], "fixed": bool(a.chs) or FING is not None}
c1, c2 = sel["c"]
n1, n2 = ("親指", "人差し指") if FING else (f"センサー{c1}", f"センサー{c2}")
l_y1, = ax[2].plot([], [], color="#d9622b", lw=1.6, label=f"{n1} 正解")
l_p1, = ax[2].plot([], [], color="#d9622b", lw=1.6, ls="--", label=f"{n1} 予測")
l_y2, = ax[2].plot([], [], color="#2ba84a", lw=1.6, label=f"{n2} 正解")
l_p2, = ax[2].plot([], [], color="#2ba84a", lw=1.6, ls="--", label=f"{n2} 予測")
ax[0].set_ylabel("筋電の大きさ\n（バンドの平均）")
axb.set_title("今の電極ごとの大きさ", fontsize=10)
ax[1].set_ylabel("力 10ch の平均 [%MVC]"); ax[1].legend(loc="upper left", fontsize=9)
ax[2].set_ylabel("力 [%MVC]"); ax[2].legend(loc="upper left", fontsize=8, ncol=2)
ax[2].set_xlabel("時間 [秒]")
for x in ax:
    x.grid(alpha=0.3)
info = ax[1].text(0.99, 0.95, "ラズパイからのデータを待っています…", transform=ax[1].transAxes, ha="right", va="top")
fig.suptitle("ラズパイでのリアルタイム推定：正解と予測")


def update(_):
    receive()
    if len(buf["t"]) < 2:
        return
    t = np.array(buf["t"]); Y = np.array(buf["y"]); P = np.array(buf["p"])
    if not sel["fixed"] and count[0] >= 250:                # 最初の 5 秒で、力の変化が大きいセンサーを 2 つ選ぶ
        sel["c"] = list(np.argsort(Y.std(0))[::-1][:2] + 1); sel["fixed"] = True
        for ln, c, kind in [(l_y1, sel["c"][0], "正解"), (l_p1, sel["c"][0], "予測"), (l_y2, sel["c"][1], "正解"), (l_p2, sel["c"][1], "予測")]:
            ln.set_label(f"センサー{c} {kind}")
        ax[2].legend(loc="upper left", fontsize=8, ncol=2)
    c1, c2 = sel["c"]
    ym, pm = Y.mean(1), P.mean(1)                              # 中段：10ch の平均
    if FING:                                                   # 指に当たるセンサーの平均
        Y = np.stack([Y[:, FING[0]].mean(1), Y[:, FING[1]].mean(1)], 1); P = np.stack([P[:, FING[0]].mean(1), P[:, FING[1]].mean(1)], 1)
        c1, c2 = 1, 2
    Rm = np.array(buf["rms"])                                      # (時間, 電極)
    nch = Rm.shape[1]
    nb = max(1, nch // 8)                                          # 8 電極ずつのバンド
    names = {3: ["肘側", "中央", "手首側"], 2: ["肘側", "手首側"]}.get(nb, [f"バンド{j + 1}" for j in range(nb)])
    if img["lines"] is None:
        img["lines"] = [ax[0].plot([], [], color=BCOL[j % 3], lw=1.5, label=f"{names[j]}（{8 * j + 1}〜{8 * j + 8}）")[0]
                        for j in range(nb)]
        ax[0].legend(loc="upper left", fontsize=8, ncol=nb)
        img["bars"] = axb.bar(np.arange(1, nch + 1), np.zeros(nch),
                              color=[BCOL[(c // 8) % 3] for c in range(nch)], width=0.8)
        axb.set_xlabel("電極の番号"); axb.set_xticks([1] + [8 * j + 8 for j in range(nb)]); axb.grid(alpha=0.3)
    band = [Rm[:, 8 * j:8 * j + 8].mean(1) for j in range(nb)]
    for ln, v in zip(img["lines"], band):
        ln.set_data(t, v)
    hi = max(v.max() for v in band); lo = min(v.min() for v in band)
    ax[0].set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
    now = Rm[-5:].mean(0)                                          # 直近 0.1 秒の平均
    img["ymax"] = max(img["ymax"] * 0.995, now.max())
    for b, v in zip(img["bars"], now):
        b.set_height(v)
    axb.set_ylim(0, img["ymax"] * 1.1)
    l_ym.set_data(t, ym); l_pm.set_data(t, pm)
    l_y1.set_data(t, Y[:, c1 - 1]); l_p1.set_data(t, P[:, c1 - 1])
    l_y2.set_data(t, Y[:, c2 - 1]); l_p2.set_data(t, P[:, c2 - 1])
    ax[2].set_xlim(t[-1] - a.window, t[-1] + 0.2)
    for x, arr in [(ax[1], np.r_[ym, pm]),
                   (ax[2], np.r_[Y[:, [c1 - 1, c2 - 1]].ravel(), P[:, [c1 - 1, c2 - 1]].ravel()])]:
        lo, hi = np.min(arr), np.max(arr)
        x.set_ylim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.1 * (hi - lo + 1e-6))
    r = np.corrcoef(ym, pm)[0, 1] if ym.std() > 0 and pm.std() > 0 else np.nan
    info.set_text(f"直近 {a.window:g} 秒の相関 r（平均の力）= {r:.2f}   受信 {count[0]} 件")


if a.save_png:
    sec, path = a.save_png.split(",")
    end = time.time() + float(sec)
    while time.time() < end:
        receive(); time.sleep(0.02)
    update(0); fig.savefig(path, dpi=110)
    print(f"受信 {count[0]} 件 → {path}")
else:
    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
