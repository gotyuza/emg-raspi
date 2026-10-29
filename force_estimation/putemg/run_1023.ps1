# 10/23 の実験（GPU の PC の emg_compare フォルダで実行）。T4 なら 1〜1.5 時間、RTX 5060 Ti ならもっと短い見込み
#   1. CNN+LSTM L：CNN固定 と 全体を微調整 の中間（BN更新 / 混ぜる層も学習）を比べる
#   2. CNN+LSTM を小さくした M・S で、全体を微調整 と CNN固定 を比べる
#   3. つまみ（親指・人差し指）だけで集計し直す ＋ 実際のエポック数
Start-Transcript -Path run_1023_log.txt -Append
python calib_compare.py --models cnnlstm --band elbow_wrist --methods 全体を微調整 CNN固定 CNN固定+BN更新 CNN固定+混ぜる層 --tag mid16
python calib_compare.py --models cnnlstm --pick cnnlstm=M --band elbow_wrist --methods 全体を微調整 CNN固定 --tag M16
python calib_compare.py --models cnnlstm --pick cnnlstm=S --band elbow_wrist --methods 全体を微調整 CNN固定 --tag S16
python finger_eval.py calib_elbow_wrist_full16.csv calib_elbow_wrist_mid16.csv calib_elbow_wrist_M16.csv calib_elbow_wrist_S16.csv
Stop-Transcript
