"""
力センサー 10 個のそれぞれが、どの指に当たるかを調べる（putemg-downloader フォルダで使う）。

調べ方: 1 日目の repeats_long で、各センサーの値と、画面に出した指ごとの目標（TRAJ_1〜4）の相関を計算し、
        一番相関が高い指をそのセンサーの指とする。
        TRAJ_1〜4 は 親指・人差し指・中指・薬指＋小指 の順（esn_putemg.py の GROUPS と同じ想定）。

使い方:
  python finger_map.py 03            → finger_map_03.json ができる
  （live_view.py / plot_replay.py に --map finger_map_03.json を付けると、親指と人差し指を表示）
"""
import json, sys
import numpy as np
import esn_putemg as E
import personal_compare as P

subj = (sys.argv[1] if len(sys.argv) > 1 else "03").zfill(2)
d1 = P.find_days("putemg_force")[subj][0]
emg, F, T, mvc = E.load(d1["repeats_long"], "elbow")
names = {"thumb": "親指", "index": "人差し指", "middle": "中指", "ringsmall": "薬指＋小指"}
out, rows = {}, []
for i in range(F.shape[1]):
    r = [np.corrcoef(F[:, i], T[:, g])[0, 1] if F[:, i].std() > 0 and T[:, g].std() > 0 else 0 for g in range(4)]
    g = int(np.nanargmax(np.abs(r)))
    out[str(i + 1)] = E.GROUPS[g]
    rows.append((i + 1, names[E.GROUPS[g]], abs(r[g]), r))
print(f"被験者 {subj}：センサーと指の対応（相関の絶対値が大きいほど確か）")
print("センサー  指          相関   （親指 / 人差し指 / 中指 / 薬指＋小指 との相関）")
for i, n, best, r in rows:
    print(f"  {i:2d}     {n:8s}  {best:.2f}   " + " / ".join(f"{x:+.2f}" for x in r))
path = f"finger_map_{subj}.json"
json.dump({"subject": subj, "sensor_to_finger": out}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("保存しました:", path)
