/*
 * emg_and_motor_combined.ino
 * ------------------------------------------------------
 * Arduino1台で以下を同時に行う:
 *   - 2ch EMGを1000Hzでサンプリングし、PCへバイナリ送信
 *   - PCからの1byteコマンドを受信し、2つのサーボモーターを制御
 *
 * 送信フレーム(PC向け, 4byte, sync含め6byte):
 *   [0]     sync1  (0xFF)
 *   [1]     sync2  (0xFE)
 *   [2-3]   ch1値  (int16, big-endian)
 *   [4-5]   ch2値  (int16, big-endian)
 *
 * 受信コマンド(PCから, 1byte):
 *   0 -> 両方のモーターをrest角度に戻す
 *   1 -> モーター1を動作角度へ、モーター2はrest角度へ
 *   2 -> モーター2を動作角度へ、モーター1はrest角度へ
 *
 * 要ライブラリ: TimerOne, Servo
 *
 * 注意: Arduino Megaでの使用を想定。Servoライブラリは標準でTimer5を
 *       使うためTimerOne(Timer1)とは競合しない想定だが、Mega以外の
 *       ボードで使う場合はタイマー競合がないか確認すること。
 * ------------------------------------------------------
 */

#include <TimerOne.h>
#include <Servo.h>

// ==================== EMG取得設定 ====================
const int CH1_PIN = A0;
const int CH2_PIN = A1;

const uint8_t SYNC1 = 0xFF;
const uint8_t SYNC2 = 0xFE;

const unsigned long SAMPLE_INTERVAL_US = 1000; // 1000Hz

volatile bool sampleReady = false;
volatile uint16_t ch1Value = 0;
volatile uint16_t ch2Value = 0;

void samplingISR() {
  // 割り込み内は最小限の処理にとどめる
  ch1Value = analogRead(CH1_PIN);
  ch2Value = analogRead(CH2_PIN);
  sampleReady = true;
}

void sendFrame(uint16_t v1, uint16_t v2) {
  Serial.write(SYNC1);
  Serial.write(SYNC2);
  // big-endianで送信(PC側 struct.unpack(">2h") 前提のため)
  Serial.write((uint8_t)(v1 >> 8));
  Serial.write((uint8_t)(v1 & 0xFF));
  Serial.write((uint8_t)(v2 >> 8));
  Serial.write((uint8_t)(v2 & 0xFF));
}

// ==================== モーター制御設定 ====================
Servo motor1;
Servo motor2;

const int MOTOR1_PIN = 9;
const int MOTOR2_PIN = 10;

// 要調整: 実際の可動範囲に合わせる
const int MOTOR1_REST_ANGLE   = 180;
const int MOTOR1_ACTIVE_ANGLE = 0;
const int MOTOR2_REST_ANGLE   = 180;
const int MOTOR2_ACTIVE_ANGLE = 0;

int motor1Target  = MOTOR1_REST_ANGLE;
int motor1Current = MOTOR1_REST_ANGLE;
int motor2Target  = MOTOR2_REST_ANGLE;
int motor2Current = MOTOR2_REST_ANGLE;

const int STEP_SIZE = 2;                  // 1ステップあたりの角度変化量(滑らかさ調整)
const unsigned long STEP_INTERVAL_MS = 15;
unsigned long lastStepTime = 0;

void stepTowards(Servo &servo, int &current, int target) {
  if (current < target) {
    current = min(current + STEP_SIZE, target);
    servo.write(current);
  } else if (current > target) {
    current = max(current - STEP_SIZE, target);
    servo.write(current);
  }
}

void handleMotorCommand(uint8_t cmd) {
  switch (cmd) {
    case 0:
      motor1Target = MOTOR1_REST_ANGLE;
      motor2Target = MOTOR2_REST_ANGLE;
      break;
    case 1:
      motor1Target = MOTOR1_ACTIVE_ANGLE;
      motor2Target = MOTOR2_REST_ANGLE;
      break;
    case 2:
      motor1Target = MOTOR1_REST_ANGLE;
      motor2Target = MOTOR2_ACTIVE_ANGLE;
      break;
    default:
      break; // 未知のコマンドは無視
  }
}

// ==================== setup / loop ====================
void setup() {
  Serial.begin(460800); // PC側のBAUDと一致させること

  Timer1.initialize(SAMPLE_INTERVAL_US);
  Timer1.attachInterrupt(samplingISR);

  motor1.attach(MOTOR1_PIN);
  motor2.attach(MOTOR2_PIN);
  motor1.write(motor1Current);
  motor2.write(motor2Current);
}

void loop() {
  // ---- EMGサンプルをPCへ送信 ----
  if (sampleReady) {
    uint16_t v1, v2;
    noInterrupts();
    v1 = ch1Value;
    v2 = ch2Value;
    sampleReady = false;
    interrupts();
    sendFrame(v1, v2);
  }

  // ---- PCからのモーターコマンド受信 ----
  if (Serial.available() > 0) {
    uint8_t cmd = Serial.read();
    handleMotorCommand(cmd);
  }

  // ---- サーボを非ブロッキングで滑らかに目標角度へ ----
  unsigned long now = millis();
  if (now - lastStepTime >= STEP_INTERVAL_MS) {
   lastStepTime = now;
    stepTowards(motor1, motor1Current, motor1Target);
    stepTowards(motor2, motor2Current, motor2Target);
 }
}
