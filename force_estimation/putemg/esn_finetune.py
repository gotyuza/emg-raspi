"""
ESN の読み出しを「前のモデルを出発点にして」学習し直す（ファインチューニング）効果を確かめる。
esn_putemg.py と同じフォルダに置いて使う（中の関数を読み込んで使う）。

ファインチューニングの式（Ridge 回帰の正則化を「0 に近づける」→「前の重み W0 に近づける」に変える）:
    W = argmin  |Y − Z W|² + λ |W − W0|²
      = (ZᵀZ + λI)⁻¹ (ZᵀY + λ W0)
  λ が大きいほど前の重みに近く、小さいほど新しいデータに合わせた答えになる。
  λ は、付けた直後のデータの最後 20% で一番誤差が小さい値を自動で選ぶ。

別の日（2 回目のセッション）での比較。全条件で、付けた直後の筋電で標準化し直す。
  A 自分の 1 回目のモデルをそのまま使う
  B 混ぜて学習し直す            （1 回目の全データ ＋ 付けた直後の正解付きデータ。今までの --recal）
  C 付けた直後のデータだけで一から学習する
  D 自分の 1 回目のモデルを出発点にファインチューニング   ← 毎日の校正を想定
  E ほかの人のモデルをそのまま使う                         ← 自分の過去データがない新しい人を想定
  F ほかの人のモデルを出発点にファインチューニング         ← 新しい人の初回の校正を想定
  （E・F では、本人の 1 回目のデータは一切使わない）

付けた直後のデータ（2 回目の repeats_long の最初の cal_sec 秒）は、前半 80% を学習、後半 20% を
λ と接触判定のしきい値の選択に使う。評価は 2 回目の sequential ＋ repeats_short。

使い方（putemg-downloader フォルダで）:
  python esn_finetune.py                         # 16ch（肘側＋手首側）、60 秒
  python esn_finetune.py --cal_sec 30
  python esn_finetune.py --band all --cal_sec 120
  python esn_finetune.py --lams 10 100 1000 10000      # λ を固定して比べる
  python esn_finetune.py --cal_task sequential          # 付けた直後のデータを「全部の指を順に押す」課題にする
"""
import argparse, sys, time
import numpy as np
import pandas as pd

import esn_putemg as E

COND = {
    "A": "A 自分の1回目そのまま",
    "B": "B 混ぜて学習し直し",
    "C": "C 直後のデータだけで一から",
    "D": "D 自分の1回目から微調整",
    "E": "E 他の人のモデルそのまま",
    "F": "F 他の人のモデルから微調整",
}
LAMS = [0.1, 1, 10, 100, 1000, 10000]


def ridge_prior(H, Y, lam, W0=None):
    """(ZᵀZ + λI) W = ZᵀY + λ W0 を解く。W0=None なら普通の Ridge（0 に近づける）"""
    Z = np.hstack([H, np.ones((len(H), 1))])
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    B = Z.T @ Y + (lam * W0 if W0 is not None else 0)
    return np.linalg.solve(A, B)


def make_model(Wc, Wf, Wb, on, off):
    m = E.TwoStage(1.0)
    m.Wc, m.Wf, m.Wb, m.on, m.off = Wc, Wf, Wb, on, off
    return m


def finetune(Hf, Yf, Hv, Yv, a, prior=None, fixed_lam=None):
    """付けた直後のデータ（学習 Hf/Yf、選択用 Hv/Yv）で読み出しを学習し直す。prior は出発点のモデル。
    fixed_lam を渡すと λ を自動で選ばずにその値を使う"""
    C = (Yf.max(1, keepdims=True) > E.CONTACT).astype(float)
    on = C[:, 0] > 0
    best = None
    for lam in ([fixed_lam] if fixed_lam is not None else LAMS):
        Wf = ridge_prior(Hf[on], Yf[on], lam, prior.Wf if prior else None)
        err = np.abs(Yv - np.clip(E.smooth(E.apply(Wf, Hv), a.smooth), 0, None))
        mv = Yv.max(1) > E.CONTACT                # 力の読み出しは、つまんでいる区間の誤差で選ぶ
        score = err[mv].mean() if mv.any() else err.mean()
        if best is None or score < best[0]:
            best = (score, lam, Wf)
    _, lam, Wf = best
    Wc = ridge_prior(Hf, C, lam, prior.Wc if prior else None)
    Wb = ridge_prior(Hf, Yf, lam, prior.Wb if prior else None)
    m = make_model(Wc, Wf, Wb, 0.6, 0.4)
    m.tune(Hv, Yv, a.smooth)                      # しきい値は学習に使っていない後半 20% で決める
    return m, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="putemg_force")
    ap.add_argument("--band", choices=list(E.BANDS), default="elbow_wrist" if "elbow_wrist" in E.BANDS else "all")
    ap.add_argument("--target", choices=["fingers"] + E.GROUPS, default="fingers")
    ap.add_argument("--cal_sec", type=float, default=60)
    ap.add_argument("--cal_task", choices=["repeats_long", "sequential"], default="repeats_long",
                    help="2 回目のどの課題の冒頭を「付けた直後のデータ」にするか（評価は残りの 2 つの課題）")
    ap.add_argument("--lams", type=float, nargs="*",
                    help="λ を固定して比べる（例: --lams 10 100 1000）。指定しなければ自動で選ぶ")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--subjects", nargs="*")
    a = ap.parse_args()

    sessions = E.find_sessions(a.root)
    if a.subjects:
        sessions = {k: v for k, v in sessions.items() if k in [s.zfill(2) for s in a.subjects]}
    need = ["repeats_long", "sequential", "repeats_short"]
    sessions = {k: v for k, v in sessions.items() if all(t in v[0] and t in v[1] for t in need)}
    if len(sessions) < 2:
        print("※ 被験者が 1 人なので、E・F（他の人のモデル）は計算しません。")
    print(f"被験者 {list(sessions)} / バンド {a.band}（{len(E.BANDS[a.band])}ch）/ 対象 {a.target}"
          f" / 付けた直後のデータ: 2 回目の {a.cal_task} の最初の {a.cal_sec:g} 秒"
          f" / λ {'固定 ' + str(a.lams) if a.lams else '自動選択'}\n")

    # ---- 1 段目：被験者ごとにデータを読み込み、1 回目のセッションでモデルを作る
    esn, D = None, {}
    for subj, (s1, s2) in sessions.items():
        t0 = time.time()
        tr = [E.prepare(s1[k], a.band, a.target) for k in ["repeats_long", "sequential"]]
        Fs, Ys = E.prepare(s1["repeats_short"], a.band, a.target)
        h = len(Fs) // 2
        Fl, Yl = E.prepare(s2[a.cal_task], a.band, a.target)
        nst = min(int(a.cal_sec * 1000 / E.HOP), len(Fl))
        te2 = [E.prepare(s2[k], a.band, a.target) for k in need if k != a.cal_task]

        Ftr = np.vstack([F for F, _ in tr])
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
        if esn is None:                           # 全員で同じリザーバー（乱数の種が同じ）を使う
            esn = E.ESN(Ftr.shape[1])

        def st(pairs, m, s):
            H, Y = [], []
            for F, y in pairs:
                H.append(esn.run((F - m) / s)[E.WASHOUT:]); Y.append(y[E.WASHOUT:])
            return np.vstack(H), np.vstack(Y)

        Htr, Ytr = st(tr, mu, sd)
        model = E.TwoStage(a.ridge); model.fit(Htr, Ytr)
        Hc, Yc = st([(Fs[:h], Ys[:h])], mu, sd); model.tune(Hc, Yc, a.smooth)
        H1, Y1 = st([(Fs[h:], Ys[h:])], mu, sd)

        # 2 回目：付けた直後の筋電で標準化し直す
        Fst = Fl[:nst]; mu2, sd2 = Fst.mean(0), Fst.std(0) + 1e-8
        Hs, Ys2 = st([(Fst, Yl[:nst])], mu2, sd2)
        k = int(len(Hs) * 0.8)
        H2, Y2 = st(te2, mu2, sd2)
        D[subj] = dict(model=model, Htr=Htr, Ytr=Ytr, H1=H1, Y1=Y1,
                       Hf=Hs[:k], Yf=Ys2[:k], Hv=Hs[k:], Yv=Ys2[k:], H2=H2, Y2=Y2)
        print(f"被験者 {subj}: 読み込みと 1 回目のモデル作成 完了（{time.time() - t0:.0f} 秒）")

    # ---- 2 段目：条件ごとに評価
    names = E.GROUPS if a.target == "fingers" else [a.target]
    rows = []

    def record(subj, cond, m, H, Y):
        P = m.predict(H, a.smooth)
        for j, n in enumerate(names):
            rows.append(dict(subj=subj, cond=cond, out=n, **E.metrics(Y[:, j], P[:, j])))

    for subj, d in D.items():
        others = [o for o in D if o != subj]
        record(subj, "同じ日（参考）", d["model"], d["H1"], d["Y1"])
        record(subj, COND["A"], d["model"], d["H2"], d["Y2"])

        mB = E.TwoStage(a.ridge)
        mB.fit(np.vstack([d["Htr"], d["Hf"]]), np.vstack([d["Ytr"], d["Yf"]]))
        mB.tune(d["Hv"], d["Yv"], a.smooth)
        record(subj, COND["B"], mB, d["H2"], d["Y2"])

        lam_list = a.lams if a.lams else [None]
        tag = lambda key, lam: COND[key] + (f" λ={lam:g}" if lam is not None else "")
        info = ""
        for lam in lam_list:
            mC, lC = finetune(d["Hf"], d["Yf"], d["Hv"], d["Yv"], a, None, lam)
            record(subj, tag("C", lam), mC, d["H2"], d["Y2"])
            mD, lD = finetune(d["Hf"], d["Yf"], d["Hv"], d["Yv"], a, d["model"], lam)
            record(subj, tag("D", lam), mD, d["H2"], d["Y2"])
            if lam is None:
                info = f"λ: C={lC:g} D={lD:g}"

        if others:   # 他の人の 1 回目のモデルを平均したものを「他の人のモデル」とする
            ms = [D[o]["model"] for o in others]
            mE = make_model(*(np.mean([getattr(m, w) for m in ms], 0) for w in ["Wc", "Wf", "Wb"]),
                            np.mean([m.on for m in ms]), np.mean([m.off for m in ms]))
            record(subj, COND["E"], mE, d["H2"], d["Y2"])
            for lam in lam_list:
                mF, lF = finetune(d["Hf"], d["Yf"], d["Hv"], d["Yv"], a, mE, lam)
                record(subj, tag("F", lam), mF, d["H2"], d["Y2"])
                if lam is None:
                    info += f" F={lF:g}"
        print(f"被験者 {subj}: 評価 完了（選ばれた {info}）")

    df = pd.DataFrame(rows)
    seen = list(dict.fromkeys(df.cond))
    order = ["同じ日（参考）"] + sorted([c for c in seen if c != "同じ日（参考）"],
                                    key=lambda c: (c[0], float(c.split("λ=")[1]) if "λ=" in c else -1))
    df["cond"] = pd.Categorical(df["cond"], order, ordered=True)
    cols = ["r", "mae", "rest", "ratio", "drop", "fs"]
    head = {"r": "r", "mae": "MAE[%MVC]", "rest": "非接触時の誤出力", "ratio": "強い力の再現率",
            "drop": "途切れ率", "fs": "誤った立ち上がり[回/分]"}
    pd.set_option("display.width", 220)

    print("\n[相関 r：条件 × 指（全被験者の平均）]")
    print(df.pivot_table(index="cond", columns="out", values="r", aggfunc="mean", observed=True)
          [names].round(2).to_string())
    print("\n[MAE %MVC：条件 × 指（全被験者の平均）]")
    print(df.pivot_table(index="cond", columns="out", values="mae", aggfunc="mean", observed=True)
          [names].round(1).to_string())
    print("\n[全項目：指と被験者で平均]")
    print(df.groupby("cond", observed=True)[cols].mean().rename(columns=head).round(2).to_string())
    print("\n[被験者ごとの相関 r（指で平均）]")
    print(df.pivot_table(index="cond", columns="subj", values="r", aggfunc="mean", observed=True)
          .round(2).to_string())


if __name__ == "__main__":
    main()
