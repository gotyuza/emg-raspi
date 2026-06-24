

#include <TimerOne.h>   //  "TimerOne" をインストール

#define NUM_CH       1
#define GRIP_CH      0   // 握力チャンネル数
#define SAMPLE_HZ    1000//サンプリング周波数
#define INTERVAL_US  (1000000 / SAMPLE_HZ)   // 1000μs

#define EMG_START_PIN  A0
#define GRIP_START_PIN (A0 + NUM_CH)

// タイマー割り込みフラグ（volatileで宣言）
volatile bool sampleReady = false;//volatileは割り込みで書き換えられる場合があることを伝える


void onTimer() {//タイマーが1000μs経過するたびにArduinoが自動でonTimer()を呼ぶ
  sampleReady = true;
}

void setup() {
  Serial.begin(460800);//460800バイト/秒までデータを送ることができる。
  //1バイト送るのに10ビット。同期バイトが2バイト、11ch×2バイトの24バイトが1000Hzで24000バイト/秒240000bpsは必要




  // タイマー1を1000μs（1000Hz）で設定
  Timer1.initialize(INTERVAL_US);
  Timer1.attachInterrupt(onTimer);//タイマーが満了するたびにonTimerを呼ぶ
}

void loop() {
  if (!sampleReady) return;
  sampleReady = false;

  // タイムスタンプ取得（マイクロ秒）
  uint32_t ts_us = micros();


  // 同期バイト（フレーム先頭の識別用）
  Serial.write((uint8_t)0xFF);
  Serial.write((uint8_t)0xFE);

  // タイムスタンプを4バイトで送信
  Serial.write((uint8_t)(ts_us >> 24));
  Serial.write((uint8_t)(ts_us >> 16));
  Serial.write((uint8_t)(ts_us >> 8));
  Serial.write((uint8_t)(ts_us));

  // A0〜A5 を順番にAD変換して送信（各ch 2バイト）
  for (int ch = 0; ch < NUM_CH; ch++) {
    int16_t val = (int16_t)analogRead(EMG_START_PIN + ch);  // 0〜1023 ch０～5を読み取る
    Serial.write((uint8_t)(val >> 8));       // 上位バイト  シリアル通信は一度に１バイトとしか送れないので2バイトを2つに分ける
    Serial.write((uint8_t)(val & 0xFF));     // 下位バイト
  }


  for (int ch = 0; ch < GRIP_CH; ch++) {
    int16_t val = (int16_t)analogRead(GRIP_START_PIN + ch);  // 0〜1023 ch０～5を読み取る
    Serial.write((uint8_t)(val >> 8));       // 上位バイト  シリアル通信は一度に１バイトとしか送れないので2バイトを2つに分ける
    Serial.write((uint8_t)(val & 0xFF));     // 下位バイト
  }
}
