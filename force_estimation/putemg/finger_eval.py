"""
calib_compare.py の結果（calib_*.csv）を、つまみ動作に関係する指だけで集計し直す。GPU 不要、すぐ終わる。

  全体     : 力センサー 10 個の平均（これまでの数字）
  つまみ   : センサー 1〜4（親指 1・2 ＋ 人差し指 3・4）
  親指     : センサー 1・2
  人差し指 : センサー 3・4
  （センサーと指の対応は putEMG-Force の装置図による）

あわせて、校正で実際に回ったエポック数と計算時間も表示する。

使い方（csv があるフォルダで）:
  python finger_eval.py calib_elbow_wrist_full16.csv calib_elbow_wrist_cnnfix16.csv
  python finger_eval.py calib_*.csv           （PowerShell ではファイル名を並べて書く）
出力: 画面の表と finger_eval.csv
"""
import sys, glob
import numpy as np
import pandas as pd

files = [f for a in sys.argv[1:] for f in (glob.glob(a) or [a])]
if not files:
    sys.exit("calib_*.csv を指定してください")
GROUPS = {"全体": range(1, 11), "つまみ(1-4)": [1, 2, 3, 4], "親指(1,2)": [1, 2], "人差し指(3,4)": [3, 4]}

d = pd.concat([pd.read_csv(f).assign(file=f) for f in files], ignore_index=True)
d["cal_min"] = d["cal_min"].astype(float)
mx = d.groupby("subj")["cal_min"].transform("max")
d["len"] = np.where(d["method"] == "同じ日（参考）", "同じ日",
                    np.where(d["cal_min"] >= mx - 1e-6, "最大", d["cal_min"].map(lambda v: f"{v:g}分")))
order = ["同じ日"] + sorted([x for x in d["len"].unique() if x not in ("同じ日", "最大")], key=lambda v: float(v[:-1])) + ["最大"]
order = [o for o in order if o in d["len"].unique()]

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
rows = []
for gname, chs in GROUPS.items():
    g = d[d["ch"].isin(list(chs))]
    # センサー → (被験者, 種) で平均してから、全体を平均（被験者ごとの重みをそろえる）
    per = g.groupby(["file", "model", "method", "len", "subj", "seed"])[["nrmse", "r"]].mean().reset_index()
    t = per.groupby(["model", "method", "len"])[["nrmse", "r"]].mean().reset_index()
    t["group"] = gname
    rows.append(t)
    print(f"\n==== {gname}：NRMSE / r（2 日目のテスト、被験者 × 種の平均。「同じ日」は 1 日目）====")
    nr = t.pivot_table(index=["model", "method"], columns="len", values="nrmse")[order].round(3)
    rr = t.pivot_table(index=["model", "method"], columns="len", values="r")[order].round(2)
    print((nr.astype(str) + " / " + rr.astype(str)).to_string())

out = pd.concat(rows)
out.to_csv("finger_eval.csv", index=False, encoding="utf-8-sig")

if "epochs" in d.columns:
    e = d.dropna(subset=["epochs"]).drop_duplicates(["file", "model", "method", "len", "subj", "seed"])
    if len(e):
        print("\n==== 校正で実際に回ったエポック数（平均）====")
        print(e.pivot_table(index=["model", "method"], columns="len", values="epochs", aggfunc="mean")
              .reindex(columns=[o for o in order if o != "同じ日"]).round(1).to_string())
        print("\n==== 校正の計算時間 [秒]（このとき使った PC / GPU での値）====")
        print(e.pivot_table(index=["model", "method"], columns="len", values="cal_sec", aggfunc="mean")
              .reindex(columns=[o for o in order if o != "同じ日"]).round(1).to_string())
print("\n保存しました: finger_eval.csv")
