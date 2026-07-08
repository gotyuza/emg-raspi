import serial
import numpy as np
import tensorflow as tf
import joblib
import time

model = tf.keras.models.load_model('/home/pi/best_cnn_model.keras')
scaler = joblib.load('/home/pi/emg_scaler.pkl')
LABEL_NAMES = {0: '安静', 1: '人差し指', 2: '中指'}
WINDOW_SIZE = 200

dummy = np.zeros((1, WINDOW_SIZE, 2), dtype=np.float32)
_ = model(dummy, training=False)
print("モデル読み込み完了")

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
print("Arduino接続完了")

buffer = []
collecting = False

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()

    if line == 'START':
        buffer = []
        collecting = True
        continue

    if collecting:
        try:
            parts = line.split(',')
            if len(parts) == 2:
                v1 = float(parts[0])
                v2 = float(parts[1])
                buffer.append([v1, v2])

                if len(buffer) == WINDOW_SIZE:
                    collecting = False
                    window = np.array(buffer, dtype=np.float32)

                    # デバッグ用
                    print(f"ch1: mean={window[:,0].mean():.3f} std={window[:,0].std():.3f}")
                    print(f"ch2: mean={window[:,1].mean():.3f} std={window[:,1].std():.3f}")

                    ser.reset_input_buffer()
                    window_scaled = scaler.transform(window)
                    inp = window_scaled.reshape(1, WINDOW_SIZE, 2)
                    result = model(inp, training=False).numpy()
                    label = int(np.argmax(result))
                    confidence = float(np.max(result))
                    print(f"予測: {LABEL_NAMES[label]} ({confidence:.2%})")
                    ser.write(f"{label}\n".encode())
        except ValueError:
            pass
