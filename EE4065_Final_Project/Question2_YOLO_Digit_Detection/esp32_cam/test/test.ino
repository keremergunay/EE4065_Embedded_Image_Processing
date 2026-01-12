/*
 * ESP32-CAM Test Kodu
 * Kamera Modülü: RHYX M21-45 (OV2640)
 * 
 * Bu kod ESP32-CAM modülünü test eder ve web sunucu üzerinden
 * kamera görüntüsünü akışla gösterir.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"

// WiFi Ayarları
const char* ssid = "Xiaomi 12T";
const char* password = "okps2644";

// ESP32-CAM AI-Thinker Pin Tanımlamaları
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_BUILTIN        4  // ESP32-CAM'deki flash LED

// Web sunucu
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// HTML sayfa içeriği
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM Test</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      text-align: center; 
      margin: 20px;
      background-color: #f0f0f0;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 { color: #333; }
    img { 
      max-width: 100%; 
      border: 2px solid #ddd;
      border-radius: 5px;
    }
    .button {
      background-color: #4CAF50;
      border: none;
      color: white;
      padding: 15px 32px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 16px;
      margin: 10px;
      cursor: pointer;
      border-radius: 5px;
    }
    .button:hover { background-color: #45a049; }
    .info {
      background: #e7f3fe;
      padding: 10px;
      border-radius: 5px;
      margin: 10px 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎥 ESP32-CAM Test Arayüzü</h1>
    <div class="info">
      <p><strong>Kamera Modülü:</strong> RHYX M21-45 (OV2640)</p>
      <p><strong>Durum:</strong> <span id="status">Bağlanıyor...</span></p>
    </div>
    
    <h2>Canlı Görüntü Akışı</h2>
    <img id="stream" src="">
    
    <div>
      <button class="button" onclick="toggleFlash()">💡 Flash Aç/Kapat</button>
      <button class="button" onclick="capturePhoto()">📸 Fotoğraf Çek</button>
      <button class="button" onclick="location.reload()">🔄 Yenile</button>
    </div>
    
    <h3>Yakalanan Fotoğraf</h3>
    <img id="photo" style="display:none;">
  </div>

  <script>
    window.onload = function() {
      document.getElementById('stream').src = window.location.href.slice(0, -1) + ':81/stream';
      document.getElementById('status').textContent = 'Bağlı ✓';
      document.getElementById('status').style.color = 'green';
    }
    
    function toggleFlash() {
      fetch('/flash')
        .then(response => response.text())
        .then(data => console.log(data));
    }
    
    function capturePhoto() {
      fetch('/capture')
        .then(response => response.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          const img = document.getElementById('photo');
          img.src = url;
          img.style.display = 'block';
        });
    }
  </script>
</body>
</html>
)rawliteral";

// HTTP stream handler
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  if(res != ESP_OK) return res;

  while(true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Kamera yakalama başarısız");
      res = ESP_FAIL;
    } else {
      if(fb->width > 400){
        if(fb->format != PIXFORMAT_JPEG){
          bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
          esp_camera_fb_return(fb);
          fb = NULL;
          if(!jpeg_converted){
            Serial.println("JPEG dönüşümü başarısız");
            res = ESP_FAIL;
          }
        } else {
          _jpg_buf_len = fb->len;
          _jpg_buf = fb->buf;
        }
      }
    }
    
    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, "\r\n--frame\r\n", 13);
    }
    
    if(fb){
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    
    if(res != ESP_OK) break;
  }
  return res;
}

// Ana sayfa handler
static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, (const char *)INDEX_HTML, strlen(INDEX_HTML));
}

// Fotoğraf çekme handler
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Fotoğraf çekme başarısız");
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  
  res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// Flash kontrol handler
bool flashState = false;
static esp_err_t flash_handler(httpd_req_t *req) {
  flashState = !flashState;
  digitalWrite(LED_BUILTIN, flashState ? HIGH : LOW);
  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_send(req, flashState ? "Flash Açık" : "Flash Kapalı", -1);
}

// Web sunucu başlatma
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = index_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t capture_uri = {
    .uri       = "/capture",
    .method    = HTTP_GET,
    .handler   = capture_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t flash_uri = {
    .uri       = "/flash",
    .method    = HTTP_GET,
    .handler   = flash_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &flash_uri);
  }

  config.server_port = 81;
  config.ctrl_port = 32768;
  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Brown-out detector devre dışı
  
  Serial.begin(115200);
  Serial.println("\n\nESP32-CAM Test Başlatılıyor...");
  
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Kamera yapılandırması
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // PSRAM var mı kontrol et
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA; // 1600x1200
    config.jpeg_quality = 10;
    config.fb_count = 2;
    Serial.println("PSRAM bulundu - Yüksek çözünürlük aktif");
  } else {
    config.frame_size = FRAMESIZE_SVGA; // 800x600
    config.jpeg_quality = 12;
    config.fb_count = 1;
    Serial.println("PSRAM bulunamadı - Standart çözünürlük aktif");
  }

  // Kamera başlatma
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamera başlatma hatası: 0x%x\n", err);
    return;
  }
  Serial.println("✓ Kamera başarıyla başlatıldı!");

  // Kamera sensör ayarları
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);     // -2 ile 2 arası
    s->set_contrast(s, 0);       // -2 ile 2 arası
    s->set_saturation(s, 0);     // -2 ile 2 arası
    s->set_sharpness(s, 0);      // -2 ile 2 arası
    s->set_special_effect(s, 0); // 0: Normal
    s->set_whitebal(s, 1);       // 0: Kapalı, 1: Açık
    s->set_awb_gain(s, 1);       // 0: Kapalı, 1: Açık
    s->set_wb_mode(s, 0);        // 0: Auto
    s->set_exposure_ctrl(s, 1);  // 0: Kapalı, 1: Açık
    s->set_aec2(s, 0);           // 0: Kapalı, 1: Açık
    s->set_gain_ctrl(s, 1);      // 0: Kapalı, 1: Açık
    s->set_agc_gain(s, 0);       // 0 ile 30 arası
    s->set_gainceiling(s, (gainceiling_t)0); // 0 ile 6 arası
    s->set_bpc(s, 0);            // 0: Kapalı, 1: Açık
    s->set_wpc(s, 1);            // 0: Kapalı, 1: Açık
    s->set_raw_gma(s, 1);        // 0: Kapalı, 1: Açık
    s->set_lenc(s, 1);           // 0: Kapalı, 1: Açık
    s->set_hmirror(s, 0);        // 0: Kapalı, 1: Açık
    s->set_vflip(s, 0);          // 0: Kapalı, 1: Açık
    s->set_dcw(s, 1);            // 0: Kapalı, 1: Açık
    s->set_colorbar(s, 0);       // 0: Kapalı, 1: Açık
    Serial.println("✓ Kamera sensör ayarları yapıldı");
  }

  // WiFi bağlantısı
  WiFi.begin(ssid, password);
  Serial.print("WiFi'ye bağlanılıyor");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n✓ WiFi bağlantısı başarılı!");
  Serial.print("IP Adresi: http://");
  Serial.println(WiFi.localIP());
  
  // Web sunucu başlat
  startCameraServer();
  
  Serial.println("\n═══════════════════════════════════════");
  Serial.println("ESP32-CAM Hazır!");
  Serial.println("═══════════════════════════════════════");
  Serial.print("Ana Sayfa: http://");
  Serial.println(WiFi.localIP());
  Serial.print("Video Akışı: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  Serial.println("═══════════════════════════════════════\n");
}

void loop() {
  delay(10000);
  Serial.print("Çalışıyor - Bellek: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" byte");
}