/*
 * ESP32 LED Blink Test
 * Upload this first to verify ESP32 toolchain works
 */

#define LED_PIN 4  // Flash LED on ESP32-CAM

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("ESP32 Ready!");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED ON");
  delay(1000);
  
  digitalWrite(LED_PIN, LOW);
  Serial.println("LED OFF");
  delay(1000);
}
