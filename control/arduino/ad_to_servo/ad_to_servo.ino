#include <Servo.h>

#define CH1_PIN A0
#define CH2_PIN A1
#define SAMPLE_RATE 1000
#define WINDOW_SIZE 200

#define SERVO1_PIN 9
#define SERVO2_PIN 10

Servo servo1;
Servo servo2;

unsigned long lastSampleTime = 0;
int sampleCount = 0;
float ch1Buffer[WINDOW_SIZE];
float ch2Buffer[WINDOW_SIZE];
int lastLabel = -1;

void setup() {
  Serial.begin(115200);
  analogReference(DEFAULT);
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo1.write(0);
  servo2.write(0);
}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleTime >= 1000) {
    lastSampleTime = now;

   float v1 = (analogRead(CH1_PIN) * 5.0 / 1023.0 - 2.5) * 2.0;
   float v2 = (analogRead(CH2_PIN) * 5.0 / 1023.0 - 2.5) * 2.0;

    ch1Buffer[sampleCount] = v1;
    ch2Buffer[sampleCount] = v2;
    sampleCount++;

    if (sampleCount >= WINDOW_SIZE) {
      sampleCount = 0;
      Serial.println("START");
      for (int i = 0; i < WINDOW_SIZE; i++) {
        Serial.print(ch1Buffer[i], 4);
        Serial.print(",");
        Serial.println(ch2Buffer[i], 4);
      }
    }
  }

  if (Serial.available()) {
    int label = Serial.parseInt();
    if (label != lastLabel) {
      lastLabel = label;
      if (label == 0) {
        servo1.write(0);
        servo2.write(0);
      } else if (label == 1) {
        servo1.write(90);
        servo2.write(0);
      } else if (label == 2) {
        servo1.write(0);
        servo2.write(90);
      }
    }
  }
}