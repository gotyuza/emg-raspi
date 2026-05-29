# control_rnn.py
# 学習済みLSTMモデルを使ってリアルタイムで義手を制御

import time
import numpy as np
import torch
import torch.nn as nn
import serial
from collections import deque

# ================================
# 設定
# ================================
MODEL_PATH  = 'models/rnn_model.pth'
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200
CONTROL_HZ  = 50        # 制御周波数（50Hz）
THRESHOLD   = 0.05      # 力の閾値
WINDOW_SIZE = 200       # LSTMの入力ウィンドウ（200ms）
INPUT_SIZE  = 8         # EMG 8チャンネル
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
OUTPUT_SIZE = 5         # 指5本
DROPOUT     = 0.2

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
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ================================
# モデル読み込み
# ================================
def load_model():
    """学習済みLSTMモデルを読み込む"""
    try:
        model = EMG_LSTM()
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu')
        )
        model.eval()
        print(f'モデル読み込み完了: {MODEL_PATH}')
        return model
    except Exception as e:
        print(f'モデル読み込み失敗: {e}')
        print('先にtrain_rnn.pyを実行してください')
        return None

# ================================
# シリアル通信（STM32）
# ================================
def init_serial():
    """STM32とのシリアル通信を初期化"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f'STM32接続完了: {SERIAL_PORT}')
        return ser
    except Exception as e:
        print(f'シリアル接続失敗: {e}')
        return None

def send_force(ser, force):
    """各指の力をSTM32へ送信"""
    data = [int(f * 255) for f in force]
    ser.write(bytes(data))

# ================================
# ADC読み取り
# ================================
def read_emg_8ch():
    """EMG 8チャンネル読み取り"""
    # TODO: Arduino/MCP3208接続後に実装
    import random
    return [random.randint(0, 1023) for _ in range(8)]

# ================================
# リアルタイム制御
# ================================
def realtime_control(model, ser):
    """リアルタイムで義手を制御"""
    print('\nリアルタイム制御開始！')
    print('Ctrl+Cで終了\n')

    interval = 1 / CONTROL_HZ  # 20ms

    # ウィンドウバッファ
    # 200ms分のデータを保持するキュー
    buffer = deque(
        [[0.0] * INPUT_SIZE] * WINDOW_SIZE,
        maxlen=WINDOW_SIZE
    )

    try:
        while True:
            loop_start = time.time()

            # 1. EMG取得
            emg = read_emg_8ch()

            # 2. 正規化（0〜1023 → -1〜1）
            emg_normalized = [(v - 512) / 512 for v in emg]

            # 3. バッファに追加
            buffer.append(emg_normalized)

            # 4. ウィンドウをテンソルに変換
            window = np.array(buffer)          # (200, 8)
            window_tensor = torch.FloatTensor(
                window
            ).unsqueeze(0)                     # (1, 200, 8)

            # 5. LSTM推論
            with torch.no_grad():
                force = model(window_tensor)
                force = force.numpy()[0]       # (5,)

            # 6. 閾値処理
            force = np.where(
                force < THRESHOLD,
                0.0,
                force
            )

            # 7. 0〜1にクリップ
            force = np.clip(force, 0.0, 1.0)

            # 8. STM32へ送信
            if ser:
                send_force(ser, force)

            # 9. デバッグ表示
            finger_names = ['親','人','中','薬','小']
            force_str = ' | '.join([
                f'{name}:{f:.2f}'
                for name, f in zip(finger_names, force)
            ])
            print(f'\r{force_str}', end='')

            # 10. 待機（50Hz制御）
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print('\n\n制御終了')
        if ser:
            send_force(ser, [0.0, 0.0, 0.0, 0.0, 0.0])
        print('全指を開いた状態で停止')

# ================================
# メイン
# ================================
if __name__ == '__main__':
    # 1. モデル読み込み
    model = load_model()
    if model is None:
        exit()

    # 2. STM32接続
    ser = init_serial()

    # 3. リアルタイム制御開始
    realtime_control(model, ser)
