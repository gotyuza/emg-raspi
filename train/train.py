# train.py
# 収集したCSVデータでESNを学習してモデルを保存

import numpy as np
import pandas as pd
import joblib
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from reservoirpy.nodes import Reservoir, Ridge

# ================================
# 設定
# ================================
DATA_DIR    = 'data/'
MODEL_DIR   = 'models/'
MODEL_NAME  = 'esn_model.pkl'

# ESNパラメータ
RESERVOIR_SIZE = 1000
SPECTRAL_RADIUS= 0.9
INPUT_SCALING  = 0.1
RIDGE          = 1e-6
WARMUP         = 100

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
# データ読み込み
# ================================
def load_all_data():
    """dataフォルダ内の全CSVを読み込む"""
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

    if len(csv_files) == 0:
        print('CSVファイルが見つかりません！')
        print(f'{DATA_DIR}にCSVファイルを置いてください')
        return None, None

    X_all = []
    Y_all = []

    for csv_file in csv_files:
        print(f'読み込み中: {csv_file}')
        df = pd.read_csv(csv_file)

        # EMGと圧力に分割
        X = df[EMG_COLS].values       # shape: (N, 8)
        Y = df[PRESSURE_COLS].values  # shape: (N, 5)

        X_all.append(X)
        Y_all.append(Y)

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)

    print(f'\n読み込み完了!')
    print(f'データ数: {len(X_all)}サンプル')
    print(f'EMG shape: {X_all.shape}')
    print(f'圧力 shape: {Y_all.shape}')

    return X_all, Y_all

# ================================
# 前処理
# ================================
def preprocess(X, Y):
    """正規化"""
    # EMG: 0〜1023 → -1〜1
    X_normalized = (X - 512) / 512

    # 圧力: 0〜1（すでに正規化済み）
    Y_normalized = np.clip(Y, 0.0, 1.0)

    return X_normalized, Y_normalized

# ================================
# ESN学習
# ================================
def train_esn(X, Y):
    """ESNを学習する"""
    print('\nESN学習開始...')

    # モデル構築
    reservoir = Reservoir(
        units          = RESERVOIR_SIZE,
        sr             = SPECTRAL_RADIUS,
        input_scaling  = INPUT_SCALING,
        rc_connectivity= 0.1
    )
    readout = Ridge(
        output_dim = 5,   # 指5本
        ridge      = RIDGE
    )
    esn = reservoir >> readout

    # 学習
    esn.fit(X, Y, warmup=WARMUP)
    print('学習完了!')

    return esn

# ================================
# モデル保存
# ================================
def save_model(esn, scaler=None):
    """学習済みモデルを保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)

    joblib.dump(esn, model_path)
    print(f'\nモデル保存完了: {model_path}')

# ================================
# 精度評価
# ================================
def evaluate(esn, X, Y):
    """学習データでの精度確認"""
    Y_pred = esn.run(X)

    # 平均絶対誤差
    mae = np.mean(np.abs(Y_pred - Y))
    print(f'\n評価結果:')
    print(f'平均絶対誤差(MAE): {mae:.4f}')

    # 各指ごとの誤差
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

    # 3. 学習
    esn = train_esn(X, Y)

    # 4. 評価
    evaluate(esn, X, Y)

    # 5. モデル保存
    save_model(esn)
