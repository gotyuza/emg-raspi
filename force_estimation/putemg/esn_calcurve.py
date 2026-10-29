"""
「校正の時間を長くすると、別の日の精度はどこまで上がるのか」を調べる（校正時間カーブ）。
esn_putemg.py・esn_finetune.py と同じフォルダに置いて使う。

  校正に使うデータ : 2 回目の sequential（全部の指を順に）→ 続けて repeats_long、の先頭から N 秒
                     （両方使い切ると約 13 分。これが今のデータでの最大）
  答え合わせ       : 2 回目の repeats_short（全部）。校正の長さを変えても同じものを使うので公平に比べられる
  校正データの分け方: 10 秒ごとのかたまりに区切り、5 つに 1 つを「λ・しきい値を選ぶ用」、残りを学習に使う
                     （後ろ 20% だけだと、そこに 1 本の指しか出てこないことがあるため）

比べる条件（全部、校正データの筋電で標準化し直す）
  C 校正データだけで一から学習          ← 教授のやり方（毎回 再学習）
  B 1 回目の全データ ＋ 校正データで学習
  D 1 回目のモデルを出発点に微調整
  参考線: 同じ日の精度 / A 1 回目のモデルをそのまま（標準化だけ最初の 60 秒でし直す）

使い方（putemg-downloader フォルダで）:
  python esn_calcurve.py                                   # 30 秒〜最大まで
  python esn_calcurve.py --secs 60 300 780                 # 調べる長さを指定（秒）
  python esn_calcurve.py --n_res 1000                      # リザーバーを大きくする（長い校正なら効くかも）
  python esn_calcurve.py --order long_first                # repeats_long を先に使う
  python esn_calcurve.py --band all                        # 24ch

結果: calcurve_<band>_N<n_res>.csv と .png
"""
import argparse, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import esn_putemg as E
import esn_finetune as FT

DEFAULT_SECS = [30, 60, 120, 240, 480, 720, 99999]     # 99999 = 使えるだけ全部


def split_blocks(n, block=500, every=5):
    """n 個の時刻を 10 秒（500 ステップ）ごとのかたまりに分け、5 つに 1 つを選択用にする"""
    block = max(50, min(block, n // 5))
    b = np.arange(n) // block
    val = (b % every) == every - 1
    if not val.any():
        val[int(n * 0.8):] = True
    return ~val, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(E.BANDS), default="elbow_wrist")
    ap.add_argument("--target", choices=["fingers"] + E.GROUPS, default="fingers")
    ap.add_argument("--secs", type=float, nargs="*", default=DEFAULT_SECS)
    ap.add_argument("--order", choices=["seq_first", "long_first"], default="seq_first")
    ap.add_argument("--n_res", type=int, default=300)
    ap.add_argument("--lams", type=float, nargs="*", help="λ の候補（初期値 0.1〜10000 から自動で選ぶ）")
    ap.add_argument("--no_bd", action="store_true", help="B・D を計算しない（速くなる）")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()
    if a.lams:
        FT.LAMS = a.lams

    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "sans-serif"]
    sessions = E.find_sessions(a.root)
    if a.subjects:
        sessions = {k: v for k, v in sessions.items() if k in [s.zfill(2) for s in a.subjects]}
    need = ["repeats_long", "sequential", "repeats_short"]
    sessions = {k: v for k, v in sessions.items() if all(t in v[0] and t in v[1] for t in need)}
    cal_tasks = ["sequential", "repeats_long"] if a.order == "seq_first" else ["repeats_long", "sequential"]
    print(f"被験者 {list(sessions)} / バンド {a.band}（{len(E.BANDS[a.band])}ch）/ リザーバー {a.n_res}"
          f" / 校正データ: 2 回目の {' → '.join(cal_tasks)} / 答え合わせ: 2 回目の repeats_short\n")

    esn, rows = None, []
    for subj, (s1, s2) in sessions.items():
        t0 = time.time()
        tr = [E.prepare(s1[k], a.band, a.target) for k in ["repeats_long", "sequential"]]
        Fs, Ys = E.prepare(s1["repeats_short"], a.band, a.target)
        cal = [E.prepare(s2[k], a.band, a.target) for k in cal_tasks]
        Fe, Ye = E.prepare(s2["repeats_short"], a.band, a.target)
        sec = lambda F: len(F) * E.HOP / 1000
        print(f"被験者 {subj}: 長さ 1 回目 long {sec(tr[0][0]) / 60:.1f}分 / seq {sec(tr[1][0]) / 60:.1f}分 / "
              f"short {sec(Fs) / 60:.1f}分 ｜ 2 回目 {cal_tasks[0]} {sec(cal[0][0]) / 60:.1f}分 / "
              f"{cal_tasks[1]} {sec(cal[1][0]) / 60:.1f}分 / short {sec(Fe) / 60:.1f}分")

        Ftr = np.vstack([F for F, _ in tr])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        if esn is None:
            esn = E.ESN(Ftr.shape[1], n_res=a.n_res)

        def st(pairs, m, s):
            H, Y = [], []
            for F, y in pairs:
                if len(F) <= E.WASHOUT:
                    continue
                H.append(esn.run((F - m) / s)[E.WASHOUT:]); Y.append(y[E.WASHOUT:])
            return np.vstack(H), np.vstack(Y)

        names = E.GROUPS if a.target == "fingers" else [a.target]

        def record(cond, m, H, Y, cal_min, lam=np.nan):
            P = m.predict(H, a.smooth)
            for j, n in enumerate(names):
                rows.append(dict(subj=subj, cond=cond, cal_min=cal_min, lam=lam, out=n,
                                 **E.metrics(Y[:, j], P[:, j])))

        # 1 回目のモデル（同じ日の参考 と B・D の材料）
        Htr, Ytr = st(tr, mu, sd)
        model = E.TwoStage(a.ridge); model.fit(Htr, Ytr)
        h = len(Fs) // 2
        Hc, Yc = st([(Fs[:h], Ys[:h])], mu, sd); model.tune(Hc, Yc, a.smooth)
        H1, Y1 = st([(Fs[h:], Ys[h:])], mu, sd)
        record("同じ日（参考）", model, H1, Y1, np.nan)

        # A：1 回目のモデルのまま、標準化だけ校正データの最初の 60 秒で
        n60 = int(60 * 1000 / E.HOP)
        F60 = np.vstack([F for F, _ in cal])[:n60]
        He, Ye_ = st([(Fe, Ye)], F60.mean(0), F60.std(0) + 1e-8)
        record("A 1回目そのまま", model, He, Ye_, np.nan)

        total = sum(len(F) for F, _ in cal)
        for s in sorted(set(min(int(x * 1000 / E.HOP), total) for x in a.secs)):
            cal_min = round(s * E.HOP / 1000 / 60, 2)
            # 先頭から s ステップ分を課題の順に切り出す
            pairs, left = [], s
            for F, y in cal:
                if left <= 0:
                    break
                pairs.append((F[:left], y[:left])); left -= len(F[:left])
            Fall = np.vstack([F for F, _ in pairs])
            mu2, sd2 = Fall.mean(0), Fall.std(0) + 1e-8
            Hs, Ys2 = st(pairs, mu2, sd2)
            fit, val = split_blocks(len(Hs))
            Hf, Yf, Hv, Yv = Hs[fit], Ys2[fit], Hs[val], Ys2[val]
            He, Ye_ = st([(Fe, Ye)], mu2, sd2)

            mC, lC = FT.finetune(Hf, Yf, Hv, Yv, a, None, None)
            record("C 校正データだけで一から", mC, He, Ye_, cal_min, lC)
            info = f"C λ={lC:g}"
            if not a.no_bd:
                mB = E.TwoStage(a.ridge)
                mB.fit(np.vstack([Htr, Hf]), np.vstack([Ytr, Yf])); mB.tune(Hv, Yv, a.smooth)
                record("B 1回目＋校正データ", mB, He, Ye_, cal_min)
                mD, lD = FT.finetune(Hf, Yf, Hv, Yv, a, model, None)
                record("D 1回目から微調整", mD, He, Ye_, cal_min, lD)
                info += f" / D λ={lD:g}"
            r_now = np.nanmean([x["r"] for x in rows[-len(names) * (1 if a.no_bd else 3):]
                                if x["cond"].startswith("C")])
            print(f"   校正 {cal_min:5.2f} 分: C の r = {r_now:.2f}   （{info}）")
        print(f"被験者 {subj}: 完了（{time.time() - t0:.0f} 秒）\n")

    df = pd.DataFrame(rows)
    tag = f"calcurve_{a.band}_N{a.n_res}"
    df.to_csv(tag + ".csv", index=False, encoding="utf-8-sig")

    # ---- 表
    pd.set_option("display.width", 220)
    ref = df[df.cal_min.isna()].groupby("cond")[["r", "mae", "lag", "rest", "fs"]].mean().round(2)
    print("[参考線（指と被験者で平均）]")
    print(ref.to_string())
    cur = df[df.cal_min.notna()]
    # 被験者ごとに「最大」の長さが違うので、最大は "最大" とまとめる
    mx = cur.groupby("subj").cal_min.transform("max")
    asked_max = max(a.secs) >= 99999
    cur = cur.assign(len_label=np.where((cur.cal_min == mx) & asked_max, "最大",
                                        cur.cal_min.map(lambda v: f"{v:g}分")))
    order = sorted(cur.len_label.unique(), key=lambda v: 1e9 if v == "最大" else float(v[:-1]))
    print("\n[校正の長さ × 条件：相関 r（指と被験者で平均）]")
    print(cur.pivot_table(index="len_label", columns="cond", values="r", aggfunc="mean")
          .reindex(order).round(2).to_string())
    print("\n[校正の長さ × 条件：MAE %MVC]")
    print(cur.pivot_table(index="len_label", columns="cond", values="mae", aggfunc="mean")
          .reindex(order).round(1).to_string())
    c = cur[cur.cond.str.startswith("C")]
    print("\n[C の詳細：長さごと（指と被験者で平均）]")
    print(c.groupby("len_label")[["cal_min", "lam", "r", "mae", "lag", "rest", "ratio", "drop", "fs"]]
          .mean().reindex(order).rename(columns={"cal_min": "実際の分", "lam": "選ばれたλ", "mae": "MAE",
                                                 "lag": "遅れms", "rest": "非接触の誤出力",
                                                 "ratio": "強い力の再現率", "drop": "途切れ率",
                                                 "fs": "誤った立ち上がり/分"}).round(2).to_string())
    print("\n[C の指ごとの r]")
    print(c.pivot_table(index="len_label", columns="out", values="r", aggfunc="mean")
          .reindex(order)[names].round(2).to_string())
    print("\n[C の被験者ごとの r]")
    print(c.pivot_table(index="len_label", columns="subj", values="r", aggfunc="mean")
          .reindex(order).round(2).to_string())

    # ---- グラフ
    g = cur.groupby(["cond", "cal_min"])[["r", "mae"]].mean().reset_index()   # 被験者が複数なら長さごとに平均
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"C": "#d9622b", "B": "#2b7bd9", "D": "#2ba84a"}
    for k, (key, lab) in enumerate([("r", "相関 r（高いほど良い）"), ("mae", "MAE [%MVC]（低いほど良い）")]):
        for cond, d in cur.groupby("cond"):
            gg = d.groupby(["subj", "cal_min"])[key].mean().reset_index()
            # 被験者ごとの最大長が違うので、順番（何番目の長さか）でそろえて平均
            gg["rank"] = gg.groupby("subj").cal_min.rank(method="first")
            m = gg.groupby("rank")[[key, "cal_min"]].mean()
            ax[k].plot(m.cal_min, m[key], "o-", color=colors[cond[0]], label=cond)
        for cond, ls in [("同じ日（参考）", "--"), ("A 1回目そのまま", ":")]:
            v = df[df.cond == cond][key].mean()
            ax[k].axhline(v, color="gray", ls=ls, label=f"{cond} {v:.2f}")
        ax[k].set_xscale("log")
        ticks = [0.5, 1, 2, 4, 8, 13]
        ax[k].set_xticks(ticks); ax[k].set_xticklabels([f"{t:g}" for t in ticks])
        ax[k].set_xlabel("校正に使った時間 [分]（対数目盛）"); ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"校正時間と別の日の精度（{a.band} {len(E.BANDS[a.band])}ch, リザーバー {a.n_res}, "
                 f"被験者 {', '.join(sessions)}）")
    plt.tight_layout()
    plt.savefig(tag + ".png", dpi=130)
    print(f"\n保存しました: {tag}.csv / {tag}.png")


if __name__ == "__main__":
    main()
