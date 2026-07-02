import serial
import time
import struct
import sys
import numpy as np
import tensorflow as tf
import joblib
from collections import deque

# ============================================================
#  コードを動かす前の準備
# ============================================================
#  Arudinoに刺す回路からの出力が0~5Vの範囲内になっていることを確認
#  ArudinoにDC電源を刺す
#  Arudino側のコードと設定項目の{BAUD,EMG_CH,GRIP_CH,FS}が一致しているか確認する
#  EMG_CHをA0からすべて刺し、その続きからGRIP_CHを刺す
# ============================================================

# ============================================================
#  設定項目
# ============================================================
PORT      = "COM3"        # ← 環境に合わせて変更
BAUD      = 460800        # ボーレート。Arduinoと一致させること
EMG_CH    = 2              # 筋電位チャンネル数
GRIP_CH   = 0               # 握力チャンネル数
FS        = 1000          # サンプリング周波数 (Hz)。Arduinoと一致させること

WINDOW_SIZE = 200          # 推論窓幅(サンプル数)。モデルの入力shapeに合わせる
HOP_SIZE    = 50           # 何サンプルごとに推論するか(スライド幅)。要調整
NUM_CLASSES = 3            # 0: rest, 1: index, 2: middle
MODEL_PATH  = r"C:\Users\hyohy\Project 試作品\final_model.keras"      # Keras形式の学習済みモデル
SCALER_PATH = r"C:\Users\hyohy\Project 試作品\emg_scaler.pkl"         # 学習時にfitしたStandardScaler(joblib形式)

ENABLE_RMS_GATE    = True   # 振幅が小さい窓は無条件でrest扱いにするか
RMS_GATE_THRESHOLD = 530.0   # 生値ベースRMS閾値。実測して調整すること

ENABLE_MOTOR_OUTPUT =True   # Trueにするとモーターコマンドを送信する
CLASS_TO_MOTOR_CMD = {0: 0, 1: 1, 2: 2}

# ---- デバッグ用設定 ----
DEBUG_PRINT_EVERY_INFER = True    # 推論のたびにRMSとlogitsを表示するか
DEBUG_PRINT_RAW_EVERY_N = 0       # 0以外にすると、生ADC値をNサンプルごとに表示(重いので通常0)
# ============================================================

total_ch   = EMG_CH + GRIP_CH
FRAME_SIZE = total_ch * 2           # 各ch 2byte(タイムスタンプなし)
SYNC1, SYNC2 = 0xFF, 0xFE           # フレーム先頭の同期バイト


def find_sync(ser: serial.Serial) -> bool:
    """ストリーム中から同期バイト 0xFF 0xFE を探す"""
    while True:
        b = ser.read(1)
        if not b:
            return False
        if b[0] == SYNC1:
            b2 = ser.read(1)
            if b2 and b2[0] == SYNC2:
                return True


# ==================== モデル読み込み(Keras) ====================
# 入力shape: (batch, WINDOW_SIZE, EMG_CH) = 時系列長が先、チャンネルが後の順
# 出力shape: (batch, NUM_CLASSES)
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def load_scaler(path):
    """学習時にfitしたStandardScalerを読み込む(joblib形式)"""
    return joblib.load(path)


def scaler_normalize(window, scaler):
    """
    window: shape (EMG_CH, WINDOW_SIZE)
    scaler: 学習時にfitしたsklearn StandardScaler。
            scaler.mean_ / scaler.scale_ はch数分の長さを想定。
    窓ごとに計算し直すのではなく、学習時に固定されたmean/stdで正規化する。
    """
    mean = scaler.mean_.reshape(-1, 1)    # (EMG_CH, 1)
    std = scaler.scale_.reshape(-1, 1)    # (EMG_CH, 1)
    return (window - mean) / std

def adc_to_voltage(window):
    """
    ADC(0～1023) → -5V～+5V
    """
    return window * (10.0 / 1023.0) - 5.0

def compute_rms(window):
    """RMSゲート用。生値ベースの全チャンネル平均RMS"""
    return float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))


def compute_ch_stats(window):
    """デバッグ用: チャンネルごとのmin/max/meanを返す"""
    stats = []
    for ch in range(window.shape[0]):
        ch_data = window[ch]
        stats.append((float(ch_data.min()), float(ch_data.max()), float(ch_data.mean())))
    return stats


def main():
    print(f"ポート      : {PORT} ({BAUD}bps)")
    print(f"サンプリング: {FS}Hz × {EMG_CH}ch (GRIP {GRIP_CH}ch)")
    print(f"推論窓幅    : {WINDOW_SIZE}サンプル / {HOP_SIZE}サンプルごとに推論")

    print("モデル読み込み中...")
    model = load_model(MODEL_PATH)

    print("標準化用scaler読み込み中...")
    scaler = load_scaler(SCALER_PATH)
    print(f"[デバッグ] scaler.mean_={scaler.mean_}, scaler.scale_={scaler.scale_}")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
    except serial.SerialException as e:
        print(f"[エラー] ポートを開けません: {e}")
        sys.exit(1)

    time.sleep(2)                  # Arduinoの自動リセット完了待ち
    ser.reset_input_buffer()       # 待機中に溜まった不要データを破棄

    # EMGチャンネルごとのリングバッファ(窓幅WINDOW_SIZE)
    ch_buffers = [deque(maxlen=WINDOW_SIZE) for _ in range(EMG_CH)]
    samples_since_infer = 0
    last_pred = 0
    last_sent_pred = None
    dropped = 0
    infer_count = 0
    t0 = time.time()

    print("リアルタイム推論を開始します(Ctrl+Cで終了)")
    try:
        while True:
            # 同期バイトを探す
            if not find_sync(ser):
                dropped += 1
                continue

            # フレームデータ読み込み(各ch 2byte × total_ch)
            raw = ser.read(FRAME_SIZE)
            if len(raw) < FRAME_SIZE:
                dropped += 1
                continue

            # 各chの値(signed 16bit, big-endian)に変換
            values = struct.unpack(f">{total_ch}h", raw)
            emg_values = values[:EMG_CH]   # GRIP_CHがある場合は末尾に含まれる

            if DEBUG_PRINT_RAW_EVERY_N and (sum(len(b) for b in ch_buffers) % DEBUG_PRINT_RAW_EVERY_N == 0):
                print(f"[デバッグ] raw emg_values={emg_values}")

            for i, v in enumerate(emg_values):
                ch_buffers[i].append(v)
            samples_since_infer += 1

            # 窓が埋まっていて、hopサンプル分新規データが溜まったら推論
            if len(ch_buffers[0]) == WINDOW_SIZE and samples_since_infer >= HOP_SIZE:
                samples_since_infer = 0
                infer_count += 1

                window = np.stack(
                    [np.array(buf, dtype=np.float32) for buf in ch_buffers],
                    axis=0
                )  # shape: (EMG_CH, WINDOW_SIZE)

                rms = compute_rms(window) if ENABLE_RMS_GATE else None

                if ENABLE_RMS_GATE and rms < RMS_GATE_THRESHOLD:
                    pred = 0  # 振幅が小さい場合はモデルを通さずrest確定
                    logits = None
                else:
                    voltage_window = adc_to_voltage(window)
                    norm_window = scaler_normalize(voltage_window, scaler)  # (EMG_CH, WINDOW_SIZE)
                    # Kerasモデルの入力shapeは (batch, WINDOW_SIZE, EMG_CH) なので転置してバッチ次元を追加
                    model_input = norm_window.T[np.newaxis, ...].astype(np.float32)
                    logits = model(model_input, training=False).numpy()  # shape: (1, NUM_CLASSES)
                    pred = int(np.argmax(logits, axis=1)[0])

                # ---- デバッグ出力: 推論のたびに毎回出す ----
                if DEBUG_PRINT_EVERY_INFER:
                    ch_stats = compute_ch_stats(window)
                    stats_str = " / ".join(
                        f"ch{i+1}[min={mn:.1f},max={mx:.1f},mean={mean:.1f}]"
                        for i, (mn, mx, mean) in enumerate(ch_stats)
                    )
                    logits_str = np.array2string(logits, precision=3) if logits is not None else "N/A(RMSゲートでスキップ)"
                    print(f"[デバッグ #{infer_count}] RMS={rms:.1f} | {stats_str} | pred={pred} | logits={logits_str}")

                if pred != last_pred:
                    rms_str = f"{rms:.1f}" if rms is not None else "N/A"
                    print(f"推定クラス変化: {last_pred} -> {pred} (RMS={rms_str})")
                last_pred = pred

                if ENABLE_MOTOR_OUTPUT:
                    cmd = CLASS_TO_MOTOR_CMD.get(pred, 0)

                    if pred != last_sent_pred:
                        ser.write(bytes([cmd]))
                        last_sent_pred = pred

    except KeyboardInterrupt:
        print("\n\nCtrl+C で終了しました。")

    finally:
        ser.close()
        elapsed = time.time() - t0
        print(f"経過時間: {elapsed:.1f}秒 / ドロップフレーム数: {dropped} / 推論回数: {infer_count}")


if __name__ == "__main__":
    main()