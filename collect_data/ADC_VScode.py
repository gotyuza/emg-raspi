

import serial
import csv
import time
import struct
import sys
from datetime import datetime
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
PORT      = "COM3"       # ← 環境に合わせて変更
BAUD      = 460800       # ボーレート（1秒当たりの転送ビット数）Arduinoと一致させること
EMG_CH    = 1            #筋電位チャンネル数
GRIP_CH   = 0             #握力チャンネル数
FS        = 1000        # サンプリング周波数 (Hz) ここもArduinoと一致させること
DURATION  = 5            # 記録時間(秒)。0=無制限 0に設定するとCtrl+Cで止めるまで無制限に記録します
FILE_NAME = "test"      #ファイルに名前を付ける 例 {FILE_NAME}_emg_20250604_153012.csv
# ============================================================
total_ch = EMG_CH + GRIP_CH
FRAME_SIZE  =4+ EMG_CH * 2 + GRIP_CH * 2        # 12バイト（同期バイトの後）1chあたり16bit(2バイト)
SYNC1, SYNC2 = 0xFF, 0xFE       #フレームの先頭を識別するための同期バイト。Arduino側から各フレームの前に2バイト送ってます


def find_sync(ser: serial.Serial) -> bool:#serというシリアル型のデータを引数にする。同期バイトを見つけたらTrueを返す関数です
    """ストリーム中から同期バイト 0xFF 0xFE を探す"""
    while True:
        b = ser.read(1)#シリアルポートから1バイトだけ読み込みます
        if not b:#空のデータが入ってきた場合Falseを返します
            return False
        if b[0] == SYNC1:#1バイト目が一致した場合
            b2 = ser.read(1)#2バイト目を読み込み
            if b2 and b2[0] == SYNC2:
                return True


def make_filename() -> str:#保存するCSVファイルの名前を現在時刻を元に生成して返します
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{FILE_NAME}_emg_{t}.csv"#例 {FILE_NAME}_emg_20250604_153012.csv


def main():
    filename = make_filename()#CSVファイルの生成
    header =( ["timestamp_s"] + [f"EMG_ch{i+1}_raw" for i in range(EMG_CH)]
             +[f"GRIP_ch{i+1}_raw" for i in range(GRIP_CH)]  
             + [f"EMG_ch{i+1}_V" for i in range(EMG_CH)]
             + [f"GRIP_ch{i+1}_V" for i in range(GRIP_CH)]#timestamp_s, ch1_raw, ch2_raw, ..., ch1_V, ch2_V, ...
    )
    print(f"ポート    : {PORT} ({BAUD}bps)")#ポートとボーレートを表示
    print(f"サンプリング: {FS}Hz × {EMG_CH}ch+{FS}Hz × {GRIP_CH}ch")#サンプリング周波数と何chかを表示
    print(f"保存先    : {filename}")#保存ファイル名を表示
    if DURATION > 0:
        print(f"記録時間  : {DURATION}秒")
    else:
        print("記録時間  : Ctrl+C で終了")
    print("-" * 30)

    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)#ポート接続に失敗した場合はエラーを表示してプログラムを終了
    except serial.SerialException as e:
        print(f"[エラー] ポートを開けません: {e}")
        sys.exit(1)#プログラムをその場で強制終了

    time.sleep(2)           # Arduinoはシリアル接続されると自動リセットされるので リセット完了待ちで2秒待機
    ser.reset_input_buffer()# リセット待ちの間に溜まったいらないデータを破棄します

    sample_count = 0#取得したサンプル数
    dropped      = 0#欠損数 欠損が多いデータは使えないし、なんか問題がある
    ts_offset    = None#Arudinoのmicro()はArudinoに電源が入った時から始まるので基準を決める用
    t0 = time.time()#記録開始時間

    try:#tryでエラーが起きたとしてもfinallyは実行されるのでポートを閉じることができる
        with open(filename, "w", newline="",encoding="utf-8") as csvfile: #"W"書き込みモード(ファイルが存在するなら上書きする)newline=""改行の自動化を無効にする。CSV.writerで改行されます。encoding="utf-8"文字コードは必要そうなら変える
            writer = csv.writer(csvfile)#ファイルオブジェクトに書き込むためのwriterオブジェクトを返します
            writer.writerow(header)#ヘッダーを一行書きます timestamp_s, ch1_raw, ch2_raw, ..., ch1_V, ch2_V, ...

            while True:#データを受信し続けるメインの無限ループ breakかCtrl+Cで抜けます
                
                elapsed = time.time() - t0# 現在時間から開始時間を引きます
                if DURATION > 0 and elapsed >= DURATION:#設定時間を超えたら終了
                    print(f"\n設定時間 {DURATION}秒 経過。記録終了。")
                    break

                # 同期バイトを探す
                if not find_sync(ser):#同期バイトが見つからなかった場合ドロップ数を増やして次のループへ
                    dropped += 1
                    continue

                # フレームデータ読み込み
                raw = ser.read(FRAME_SIZE)#同期バイトの直後に続く12バイトのフレームデータを読み込みます
                if len(raw) < FRAME_SIZE:#途中で通信が切れた場合そのフレームを捨てて次のループへ
                    dropped += 1
                    continue


                # タイムスタンプ（上位4バイト, uint32, μs単位）を復元
                ts_us = struct.unpack(">I", raw[0:4])[0]
 
                # 最初のフレームを0秒基準にする
                if ts_offset is None:
                    ts_offset = ts_us
                ts_s = round((ts_us - ts_offset) / 1_000_000, 6)#記録開始からの経過時間（秒） 小数点6桁表示


                # ビッグエンディアン(先に上位バイトが並んでいる方式 Arduinoが先に上位バイトを送ってきます) "h"符号付き16bit 6ch に変換 12バイトが6つの整数に変換されます
                # 最初の４バイト以降
                values = struct.unpack(f">{total_ch}h", raw[4:])

                # 電圧に変換
                voltages = [round((v * 5.0 / 1023.0 -2.5)*2, 6) for v in values]#小数点以下6桁 ov~5vの範囲を-5v~5vの範囲に変換

                #ts = round(elapsed, 6)#タイムスタンプを小数点以下6桁で記録 time.time()が15msでしか更新されないことが分かったので使いません
                writer.writerow([ts_s] + list(values) + voltages)#タイムスタンプ、生値数ch、生値から変換した電圧値数chをCSVに書き込み

                sample_count += 1#サンプル数プラス１

                # 進捗表示（1秒ごと）
                if sample_count % FS == 0:
                    csvfile.flush()#1秒ごとにファイルに書き込み 記録中にクラッシュとか起きた場合でもデータが1秒ごとに残る。
                    sec = sample_count // FS
                    print(f"  {sec:5d}秒経過  {sample_count:7d}サンプル  "
                          f"ドロップ: {dropped}  "
                          f"実効レート: {sample_count/elapsed:.1f}Hz", end="\r")#進捗を上書き表示

    except KeyboardInterrupt:#エラーが起きた時の処理だがCtrl+cで実行を途中でやめたときの処理
        print("\n\nCtrl+C で終了しました。")

    finally:
        ser.close()#シリアルポートを閉じます
        elapsed = time.time() - t0#時間計算
        print(f"\n保存完了: {filename}")
        print(f"  総サンプル数 : {sample_count}")
        print(f"  記録時間     : {elapsed:.2f}秒")
        print(f"  実効レート   : {sample_count/elapsed:.2f}Hz" if elapsed > 0 else "")
        print(f"  ドロップ数   : {dropped}")


if __name__ == "__main__":#このファイルが直接実行されたときだけmain()を呼ぶ
    main()