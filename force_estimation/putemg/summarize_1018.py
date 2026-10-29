"""run_1018.ps1 の結果を一覧にする（emg_compare フォルダで実行）: python summarize_1018.py"""
import csv, os
import numpy as np

for s in ["03", "04", "05"]:
    for folder, tag, label in [("export24", "", "24ch"), ("export16", "_elbow_wrist", "16ch")]:
        m = os.path.join(folder, f"model_transformer_S_{s}.npz")
        c = f"importance_transformer_S_{s}{tag}.csv"
        if not os.path.exists(m):
            print(f"被験者{s} {label}: モデルなし"); continue
        B = np.load(m, allow_pickle=True)
        print(f"\n被験者{s} {label}: NRMSE {float(B['nrmse_pc']):.3f}  r {float(B['r_pc']):.2f}")
        if not os.path.exists(c):
            print("  重要度 csv なし"); continue
        rows = list(csv.reader(open(c, encoding="utf-8-sig")))
        head, rows = rows[0], rows[1:]
        el = [r for r in rows if r[0].startswith("電極")]
        bd = [r for r in rows if "バンド" in r[0]]
        for j, name in enumerate(["全体", "親指", "人差し指"]):
            top = sorted(el, key=lambda r: -float(r[j + 1]))[:5]
            print(f"  {name:6s} 上位: " + ", ".join(f"{r[0].replace('電極', '')}({float(r[j + 1]):.0f}%)" for r in top))
        print("  バンド丸ごと（全体/親指/人差し指）: " + "  ".join(f"{r[0].replace('バンド全体', '')} " + "/".join(f"{float(x):.0f}%" for x in r[1:]) for r in bd))
