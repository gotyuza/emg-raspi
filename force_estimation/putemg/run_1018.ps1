# 10/18 の実験（GPU の PC の emg_compare フォルダで実行）
#   1. 被験者 03・04・05 の Transformer S を 24ch と 16ch（肘側＋手首側）で学習して書き出す
#   2. それぞれ電極ごとの重要度を計算（並べ替え 3 回の平均）
#   ログは run_1018_log.txt にも残る
Start-Transcript -Path run_1018_log.txt -Append
foreach ($s in "03", "04", "05") {
    python export_model.py --model transformer --size S --subject $s --out export24
    python electrode_importance.py export24/model_transformer_S_$s.npz --repeats 3
    python export_model.py --model transformer --size S --subject $s --band elbow_wrist --out export16
    python electrode_importance.py export16/model_transformer_S_$s.npz --repeats 3
}
Stop-Transcript
