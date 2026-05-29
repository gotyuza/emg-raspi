# control.py
# 学習済みESNモデルを使ってリアルタイムで義手を制御

import time
import joblib
import numpy as np
import serial

# ================================
# 設定
# ================================
MODEL_PATH  = 'models/esn_model.pkl'
SERIAL_PORT = '/dev/ttyUSB0'  # ラズパイのシリアルポート
BAUD_RATE   = 115200
CONTROL_HZ  = 50              # 制御周波数（50Hz）
THRESHOLD   = 0.05            # 力の閾値（これ以下はモーターOFF）

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
        print('STM32が接続されているか確認してください')
        return None

def send_force(ser, force):
    """各指の力をSTM32へ送信"""
    # force: [親指, 人差し指, 中指, 薬指, 小指] 各0〜1
    # 0〜255に変換して送信
    data = [int(f * 255) for f in force]
    ser.write(bytes(data))  # 5バイト送信

# ================================
# ADC読み取り
# ================================
def read_emg_8ch():
    """EMG 8チャンネル読み取り"""
    # TODO: Arduino/MCP3208接続後に実装
    import random
    return [random.randint(0, 1023) for _ in range(8)]

# ================================
# モデル読み込み
# ================================
def load_model():
    """学習済みESNモデルを読み込む"""
    try:
        esn = joblib.load(MODEL_PATH)
        print(f'モデル読み込み完了: {MODEL_PATH}')
        return esn
    except Exception as e:
        print(f'モデル読み込み失敗: {e}')
        print('先にtrain.pyを実行してモデルを作成してください')
        return None

# ================================
# リアルタイム制御
# ================================
def realtime_control(esn, ser):
    """リアルタイムで義手を制御"""
    print('\nリアルタイム制御開始！')
    print('Ctrl+Cで終了\n')

    interval = 1 / CONTROL_HZ  # 20ms

    try:
        while True:
            loop_start = time.time()

            # 1. EMG取得
            emg = read_emg_8ch()

            # 2. 正規化（0〜1023 → -1〜1）
            emg_normalized = np.array(
                [(v - 512) / 512 for v in emg]
            ).reshape(1, -1)

            # 3. ESN推論
            force = esn.run(emg_normalized)[0]
            # → [親指, 人差し指, 中指, 薬指, 小指] 各0〜1

            # 4. 閾値処理
            # 閾値以下はモーターOFF（ばねで開く）
            force = np.where(
                force < THRESHOLD,
                0.0,
                force
            )

            # 5. 0〜1にクリップ
            force = np.clip(force, 0.0, 1.0)

            # 6. STM32へ送信
            if ser:
                send_force(ser, force)

            # 7. デバッグ表示
            finger_names = ['親','人','中','薬','小']
            force_str = ' | '.join([
                f'{name}:{f:.2f}'
                for name, f in zip(finger_names, force)
            ])
            print(f'\r{force_str}', end='')

            # 8. 待機（50Hz制御）
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        print('\n\n制御終了')
        # 全指を開く（力=0）
        if ser:
            send_force(ser, [0.0, 0.0, 0.0, 0.0, 0.0])
        print('全指を開いた状態で停止')

# ================================
# メイン
# ================================
if __name__ == '__main__':
    # 1. モデル読み込み
    esn = load_model()
    if esn is None:
        exit()

    # 2. STM32接続
    ser = init_serial()

    # 3. リアルタイム制御開始
    realtime_control(esn, ser)
