# force_estimation — 筋電による指の力（把持力）推定

筋電（EMG）から指先の力をリアルタイムに推定するモデルの比較・校正実験と、ラズパイ5で動かすためのコード。
公開データ putEMG-Force（被験者 03・04・05）で検証した。

## フォルダ

| フォルダ | 実行する場所 | 中身 |
|---|---|---|
| `putemg/` | PC（GPU があると速い）/ Colab | データのダウンロード、モデル比較、校正実験、ラズパイ用の書き出し |
| `pi/` | ラズパイ5 | 録画データを流して推論の精度・時間を測る、校正（再学習）の時間を測る |
| `viz/` | PC | ラズパイから送られた推定結果のリアルタイム表示、再生データのグラフ |
| `forceband/` | PC | 初期に試した ForceBand データでのベースライン（参考） |

`putemg/` のスクリプトは互いに読み込み合う（`esn_putemg.py`・`personal_compare.py` など）ので、同じフォルダに置いたまま使う。

## 使い方の流れ

```
# 1. データ（putEMG-Force）を取得 → putemg/putemg_force/ に保存
cd force_estimation/putemg
python put_force_dl.py 03 04 05

# 2. モデルの比較（同じ日）と、付け直し後の校正の比較
python personal_compare.py --band elbow_wrist --day2
python calib_compare.py --band elbow_wrist

# 3. ラズパイ用に書き出し（ONNX ＋ 設定 npz ＋ 再生データ）
python export_model.py --model cnnlstm --size L --subject 03 --band elbow_wrist --out export16

# 4. export16/ をラズパイに送り、再生して精度・処理時間を確認（ラズパイ側、pi/ のスクリプトと同じ場所で）
python pi_replay.py export16/model_cnnlstm_L_03.npz --realtime --udp <PCのIP>:5005
#    PC 側（force_estimation/ で）：リアルタイム表示
python viz/live_view.py --map putemg/finger_map_device.json
```

主なスクリプト

| スクリプト | 内容 |
|---|---|
| `putemg/personal_compare.py` | ridge / ESN / LSTM / CNN+LSTM / Transformer を、大きさ S〜XL で比較（ハイパーパラメータ試行回数は全モデル同じ） |
| `putemg/calib_compare.py` | 付け直した 2 日目の校正（なし / 一から / 全体を微調整 / 出力層だけ / 入力層だけ / CNN固定 など）を校正の長さ 30 秒〜最大で比較 |
| `putemg/export_model.py` | 学習してラズパイ用に書き出し（transformer / lstm / cnnlstm / esn / ridge） |
| `putemg/electrode_importance.py` | 電極ごとの重要度（並べ替え重要度） |
| `putemg/finger_eval.py` | 結果を親指（センサー 1・2）・人差し指（3・4）だけで集計し直す |
| `pi/pi_replay.py` | 録画した筋電を 20 ms ごとに流し、本番と同じ処理で推論（ONNX Runtime） |
| `pi/bench_calib_pi.py` | ラズパイでの校正（再学習）の時間を測る |
| `pi/bench_pi5.py` | ラズパイでの推論速度のベンチマーク（PyTorch / ONNX / int8） |
| `viz/live_view.py` | ラズパイから UDP で届く推定結果をリアルタイム表示（`--map` で親指・人差し指） |

共通の前処理：1 kHz、20〜450 Hz 帯域通過＋50 Hz ノッチ、150 ms 窓を 20 ms ごと（特徴量は電極ごとの log RMS・log WL。CNN+LSTM は波形そのもの）。出力は力センサー 10 個 [%MVC]。

## これまでの主な結果（16ch：肘側 1〜8 ＋ 手首側 17〜24）

- 同じ日のテストでは、どのモデルも大差なし（NRMSE：Transformer S 0.162、ESN 0.175 前後、ridge 0.181。24ch）。モデルを大きくしても精度は上がらない
- 24ch → 16ch の精度低下は約 3%
- 付け直した 2 日目は、校正なしでは使えない（r 0.2〜0.4）
- 4 分の校正で：CNN+LSTM L 全体を微調整 NRMSE 0.146 / r 0.78、CNN固定 0.163 / 0.70、ESN を一から 0.182 / 0.69（ESN は 11.4 分で 0.176 / 0.74）
- CNN固定＋BatchNorm 更新はほぼ効果なし。CNN固定＋電極を組み合わせる層は一部の被験者（04・05）で効果あり
- ラズパイ5：CNN+LSTM L の推論 0.71 ms（20 ms ごと、約 1 万回で遅れ 0 回、PC と同じ答え）。校正の再学習は CNN固定で約 20 秒、全体を微調整で 5〜10 分（見込み）

現時点の確定案（v1）：CNN+LSTM L・16ch。付け直し時は 4 分記録 → CNN固定で約 20 秒学習して使い始め、裏で全体を微調整して差し替える。
詳細は `docs/report_1023.md`、遅れの目標（100 ms 以内）の根拠は `docs/latency_target.md`。

## 注意

- 電極を体に付けて試すときは、PiEEG とつながっている機器をすべてバッテリー駆動にし、コンセントから切り離す
- `putemg_force/`・`cache/`・`export*/`・結果の csv/png は `.gitignore` で除外している
