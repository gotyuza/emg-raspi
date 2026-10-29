# 10/21 の実験（GPU の PC の emg_compare フォルダで実行）。全体で 1 時間前後の見込み
#   1. CNN+LSTM の校正：一から / 全体を微調整 / CNN固定（新しい方法）を 16ch で比べる（L と M の 2 サイズ）
#   2. ラズパイ用に CNN+LSTM L を書き出す（被験者 03・04・05、16ch）→ export16 フォルダ
#   3. 校正で実際に何エポック回ったかを表示
Start-Transcript -Path run_1021_log.txt -Append
python calib_compare.py --models cnnlstm --band elbow_wrist --methods 一から 全体を微調整 CNN固定 --tag cnnfix16
python calib_compare.py --models cnnlstm --pick cnnlstm=M --band elbow_wrist --methods 一から 全体を微調整 CNN固定 --tag cnnfixM16
foreach ($s in "03", "04", "05") {
    python export_model.py --model cnnlstm --size L --subject $s --band elbow_wrist --out export16
}
python -c "import pandas as pd; d=pd.concat([pd.read_csv(f) for f in ['calib_elbow_wrist_full16.csv','calib_elbow_wrist_cnnfix16.csv','calib_elbow_wrist_cnnfixM16.csv']]); print(d.dropna(subset=['epochs']).groupby(['model','method'])['epochs'].mean().round(1).to_string())"
Stop-Transcript
