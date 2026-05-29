# collect_data.py
# EMG + 圧力センサのデータを同時収集してCSVに保存

import time
import csv
from datetime import datetime

# ================================
# 設定
# ================================
SAMPLE_RATE = 1000  # Hz
DURATION    = 3     # 秒
NUM_EMG_CH  = 8     # EMGチャンネル数
NUM_PRESSURE= 5     # 圧力センサ数

# 収集するパターン
PATTERNS = [
    'rest',           # 待機
    'thumb_weak',     # 親指・弱
    'thumb_mid',      # 親指・中
    'thumb_strong',   # 親指・強
    'index_weak',     # 人差し指・弱
    'index_mid',      # 人差し指・中
    'index_strong',   # 人差し指・強
    'middle_weak',    # 中指・弱
    'middle_mid',     # 中指・中
    'middle_strong',  # 中指・強
    'ring_weak',      # 薬指・弱
    'ring_mid',       # 薬指・中
    'ring_strong',    # 薬指・強
    'little_weak',    # 小指・弱
    'little_mid',     # 小指・中
    'little_strong',  # 小指・強
    'grasp_weak',     # 全握り・弱
    'grasp_mid',      # 全握り・中
    'grasp_strong',   # 全握り・強
    'pinch_weak',     # ピンチ・弱
    'pinch_mid',      # ピンチ・中
    'pinch_strong',   # ピンチ・強
]

# ================================
# ADC読み取り（仮実装）
# 実際のADC接続後に変更してください
# ================================
def read_emg_8ch():
    """EMG 8チャンネル読み取り"""
    # TODO: Arduino/MCP3208接続後に実装
    # 仮のダミーデータを返す
    import random
    return [random.randint(0, 1023) for _ in range(NUM_EMG_CH)]

def read_pressure_5ch():
    """圧力センサ 5チャンネル読み取り"""
    # TODO: ADS1115接続後に実装
    # 仮のダミーデータを返す
    import random
    return [random.uniform(0.0, 1.0) for _ in range(NUM_PRESSURE)]

# ================================
# データ収集
# ================================
def collect_session(user_id, session_num):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'data/user{user_id}_session{session_num}_{timestamp}.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # ヘッダー
        writer.writerow([
            'time_ms', 'pattern',
            'emg_ch1','emg_ch2','emg_ch3','emg_ch4',
            'emg_ch5','emg_ch6','emg_ch7','emg_ch8',
            'pressure_thumb','pressure_index',
            'pressure_middle','pressure_ring',
            'pressure_little'
        ])

        for pattern in PATTERNS:
            print(f'\n=== {pattern} ===')
            print('準備してください...')
            time.sleep(2)
            print('動作開始！')

            time_ms = 0
            start   = time.time()

            while time.time() - start < DURATION:
                loop_start = time.time()

                # EMG + 圧力を取得
                emg      = read_emg_8ch()
                pressure = read_pressure_5ch()

                # CSV書き込み
                writer.writerow(
                    [time_ms, pattern] + emg + pressure
                )

                time_ms += 1

                # 1msになるよう待機
                elapsed = time.time() - loop_start
                if elapsed < 1/SAMPLE_RATE:
                    time.sleep(1/SAMPLE_RATE - elapsed)

            print('休憩...')
            time.sleep(2)

    print(f'\n保存完了: {filename}')

# ================================
# メイン
# ================================
if __name__ == '__main__':
    user_id     = input('ユーザーIDを入力（例: 001）: ')
    session_num = input('セッション番号を入力（例: 1）: ')
    collect_session(user_id, session_num)
