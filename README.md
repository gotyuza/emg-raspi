# emg-raspi
## 編集者
-後藤優真（管理者）  
-下村彪悟  

## フォルダ構成
- `collect_data/` : 筋電の収集（Arduino ADC）
- `train/` : 動作分類モデル（ESN / RNN）の学習
- `control/` : リアルタイム制御（推論 → Arduino / STM32）
- `force_estimation/` : 筋電による指の力（把持力）の推定。putEMG-Force でのモデル比較・校正実験、ラズパイ5での推論・校正（詳細は `force_estimation/README.md`）
- `docs/` : 報告書・資料（作業報告、遅れの目標の根拠、アクティブ電極の作り方）
