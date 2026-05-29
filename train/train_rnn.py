# train_rnn.py
# 収集したCSVデータでLSTMを学習してモデルを保存

import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ================================
# 設定
# ================================
DATA_DIR    = 'data/'
MODEL_DIR   = 'models/'
MODEL_NAME  = 'rnn_model.pth'

# LSTMパラメータ
INPUT_SIZE  = 8      # EMG 8チャンネル
HIDDEN_SIZE = 128    # 隠れ層サイズ
NUM_LAYERS  = 2      # LSTM層数
OUTPUT_SIZE = 5      # 指5本
DROPOUT     = 0.2    # ドロップアウト

# 学習パラメータ
WINDOW_SIZE = 200    # 200ms分のデータ
STRIDE      = 50     # 50msごとにスライド
BATCH_SIZE  = 32
EPOCHS      = 100
LEARNING_RATE= 0.001

# EMG・圧力のカラム名
EMG_COLS = [
    'emg_ch1','emg_ch2','emg_ch3','emg_ch4',
    'emg_ch5','emg_ch6','emg_ch7','emg_ch8'
]
PRESSURE_COLS = [
    'pressure_thumb','pressure_index',
    'pressure_middle','pressure_ring',
    'pressure_little'
]

# ================================
# LSTMモデル定義
# ================================
class EMG_LSTM(nn.Module):
    def __init__(self):
        super(EMG_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            batch_first = True,
            dropout     = DROPOUT
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_SIZE),
            nn.Sigmoid()  # 0〜1に正規化
        )

    def forward(self, x):
        # x: (batch, window, 8)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 最後の時刻の出力
        return out

# ================================
# データ読み込み
# ================================
def load_all_data():
    """dataフォルダ内の全CSVを読み込む"""
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

    if len(csv_files) == 0:
        print('CSVファイルが見つかりません！')
        return None, None

    X_all = []
    Y_all = []

    for csv_file in csv_files:
        print(f'読み込み中: {csv_file}')
        df = pd.read_csv(csv_file)

        X = df[EMG_COLS].values       # shape: (N, 8)
        Y = df[PRESSURE_COLS].values  # shape: (N, 5)

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print(f'データ数: {len(X_all)}サンプル')
    return X_all, Y_all

# ================================
# 前処理
# ================================
def preprocess(X, Y):
    """正規化"""
    X_normalized = (X - 512) / 512   # -1〜1
    Y_normalized = np.clip(Y, 0.0, 1.0)
    return X_normalized, Y_normalized

# ================================
# ウィンドウ処理
# ================================
def create_windows(X, Y):
    """時系列データをウィンドウに切り出す"""
    X_windows = []
    Y_windows = []

    for i in range(0, len(X) - WINDOW_SIZE, STRIDE):
        X_windows.append(X[i:i+WINDOW_SIZE])  # (200, 8)
        Y_windows.append(Y[i+WINDOW_SIZE-1])  # 最後の時刻のラベル

    X_windows = np.array(X_windows)  # (N, 200, 8)
    Y_windows = np.array(Y_windows)  # (N, 5)

    print(f'ウィンドウ数: {len(X_windows)}')
    return X_windows, Y_windows

# ================================
# LSTM学習
# ================================
def train_lstm(X, Y):
    """LSTMを学習する"""
    print('\nLSTM学習開始...')

    # PyTorchテンソルに変換
    X_tensor = torch.FloatTensor(X)  # (N, 200, 8)
    Y_tensor = torch.FloatTensor(Y)  # (N, 5)

    # データローダー
    dataset    = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # モデル・損失関数・最適化
    model     = EMG_LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 学習ループ
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}] '
                  f'Loss: {avg_loss:.6f}')

    print('学習完了!')
    return model

# ================================
# モデル保存
# ================================
def save_model(model):
    """学習済みモデルを保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print(f'\nモデル保存完了: {model_path}')

# ================================
# 精度評価
# ================================
def evaluate(model, X, Y):
    """精度確認"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        Y_pred   = model(X_tensor).numpy()

    mae = np.mean(np.abs(Y_pred - Y))
    print(f'\n評価結果:')
    print(f'平均絶対誤差(MAE): {mae:.4f}')

    finger_names = ['親指','人差し指','中指','薬指','小指']
    for i, name in enumerate(finger_names):
        finger_mae = np.mean(np.abs(Y_pred[:, i] - Y[:, i]))
        print(f'{name}: MAE = {finger_mae:.4f}')

# ================================
# メイン
# ================================
if __name__ == '__main__':
    # 1. データ読み込み
    X, Y = load_all_data()
    if X is None:
        exit()

    # 2. 前処理
    X, Y = preprocess(X, Y)

    # 3. ウィンドウ処理
    X, Y = create_windows(X, Y)

    # 4. 学習
    model = train_lstm(X, Y)

    # 5. 評価
    evaluate(model, X, Y)

    # 6. モデル保存
    save_model(model)
