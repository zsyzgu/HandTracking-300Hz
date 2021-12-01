/*
 * VDD ---------------------- 3.3V
 * SDA ----------------------- A4
 * SCL ----------------------- A5
 * GND ---------------------- GND
 */
#include "MPU9250.h"

MPU9250 mpu = MPU9250();
long t;
int16_t *ax, *ay, *az, *gx, *gy, *gz;
byte buf[12];

void setup(void) {
  Serial.begin(250000);  
  uint8_t temp = mpu.begin();
  mpu.set_accel_range(RANGE_8G);
  mpu.set_gyro_range(RANGE_GYRO_500);
  ax = (int16_t*)&buf[0];
  ay = (int16_t*)&buf[2];
  az = (int16_t*)&buf[4];
  gx = (int16_t*)&buf[6];
  gy = (int16_t*)&buf[8];
  gz = (int16_t*)&buf[10];
  pinMode(13, OUTPUT);
  delay(1000);
}

void loop() {
  t = micros();
  mpu.get_data((int8_t*)ax);
//  mpu.get_accel();
//  *ax = mpu.x;
//  *ay = mpu.y;
//  *az = mpu.z;
//  mpu.get_gyro();
//  *gx = mpu.gx;
//  *gy = mpu.gy;
//  *gz = mpu.gz;
  Serial.write(buf, 12);
//  Serial.print((*ax)/4096.0);
//  Serial.print(",");
//  Serial.print((*ay)/4096.0);
//  Serial.print(",");
//  Serial.println((*az)/4096.0);
  while (micros() - t < 825);
}
