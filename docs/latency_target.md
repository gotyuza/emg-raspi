# 遅れの目標を「100 ms 以内」とした理由

義手班 筋電による把持力推定　2026-10-23

## 結論

筋肉に力を入れようとしてから義手の指先に力が出るまでの遅れの目標を、**100 ms 以内、できるだけ短く**とする。

## 根拠

**Farrell & Weir（2007）**：健常者 20 人が、筋電義手として操作できる装置で Box and Block テスト（積み木を時間内にいくつ運べるか）を行った。制御の遅れを約 0〜300 ms の 7 段階、手の速さを 2 種類に変えて比べた。

- 最適な遅れは、手の動きが速い義手で約 100 ms、遅い義手で約 125 ms
- 遅れが増えるほど、成績は直線的に低下した

**Smith ら（2011）**：筋電制御の解析窓の長さを変えて調べ、最適な遅れは 88〜138 ms で、Farrell & Weir の結果とよく一致した。

## 125 ms ではなく 100 ms にした理由

- 遅れが増えるほど成績は直線的に下がり、「ここまでなら影響なし」という境目はない。そのため厳しい側を目標にした。
- 先行研究の「最適」は、判定の正確さを得るために遅れを許した兼ね合いの点である。私たちは力を連続で推定する方式で、遅れを許す理由が少ない。
- つまむ動作は細かい力の加減が必要で、遅れの影響を受けやすいと考えられる。

## 注意点

先行研究は健常者・物を運ぶ課題での結果で、つまむ動作で同じ値になるかは確認されていない。最終的には実際の操作で確かめる。

## 参考文献

1. Farrell TR, Weir RF. The Optimal Controller Delay for Myoelectric Prostheses. *IEEE Trans. Neural Syst. Rehabil. Eng.*, 15(1):111–118, 2007. https://doi.org/10.1109/tnsre.2007.891391
2. Smith LH, Hargrove LJ, Lock BA, Kuiken TA. Determining the optimal window length for pattern recognition-based myoelectric control. *IEEE Trans. Neural Syst. Rehabil. Eng.*, 19(2):186–192, 2011. https://pmc.ncbi.nlm.nih.gov/articles/PMC4241762
