/*
 * ESP32-CAM Web Server - Image Capture
 * SD kart yerine WiFi web server ile görüntü paylaşımı
 * 
 * Board: AI Thinker ESP32-CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ==================== WiFi AYARLARI ====================
// Kendi WiFi bilgilerinizi girin!
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ==================== Web Server ====================
WebServer server(80);

// ==================== ESP32-CAM AI-Thinker Pin Tanımları ====================
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
#define FLASH_GPIO_NUM     4

// ==================== KAMERA AYARLARI ====================
bool initCamera() {
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
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 10;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Kamera baslatma hatasi: 0x%x\n", err);
        return false;
    }
    return true;
}

// ==================== WEB SAYFASI ====================
void handleRoot() {
    String html = "<!DOCTYPE html><html><head>";
    html += "<meta charset='UTF-8'>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<title>ESP32-CAM Capture</title>";
    html += "<style>";
    html += "body{font-family:Arial,sans-serif;text-align:center;background:#1a1a2e;color:#fff;margin:0;padding:20px;}";
    html += "h1{color:#00d4ff;}";
    html += ".btn{background:linear-gradient(45deg,#00d4ff,#0099cc);color:#fff;padding:15px 30px;";
    html += "border:none;border-radius:25px;font-size:18px;cursor:pointer;margin:10px;text-decoration:none;display:inline-block;}";
    html += ".btn:hover{background:linear-gradient(45deg,#0099cc,#00d4ff);}";
    html += "img{max-width:100%;border-radius:10px;margin-top:20px;box-shadow:0 4px 15px rgba(0,212,255,0.3);}";
    html += ".info{background:#16213e;padding:15px;border-radius:10px;margin:20px auto;max-width:500px;}";
    html += "</style></head><body>";
    html += "<h1>ESP32-CAM Web Server</h1>";
    html += "<div class='info'>";
    html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
    html += "</div>";
    html += "<a href='/capture' class='btn'>Fotograf Cek</a>";
    html += "<a href='/stream' class='btn'>Canli Izle</a>";
    html += "<br><br>";
    html += "<img id='photo' src='/capture' onerror='this.style.display=\"none\"'>";
    html += "<script>";
    html += "function refresh(){document.getElementById('photo').src='/capture?'+Date.now();}";
    html += "</script>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
}

// ==================== JPEG CAPTURE ====================
void handleCapture() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Capture failed!");
        return;
    }
    
    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
    
    esp_camera_fb_return(fb);
    Serial.println("Capture sent!");
}

// ==================== MJPEG STREAM ====================
void handleStream() {
    WiFiClient client = server.client();
    
    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    client.print(response);
    
    while (client.connected()) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Capture failed");
            break;
        }
        
        client.printf("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", fb->len);
        client.write(fb->buf, fb->len);
        client.print("\r\n");
        
        esp_camera_fb_return(fb);
        
        if (!client.connected()) break;
        delay(33);  // ~30 FPS
    }
}

// ==================== SETUP ====================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n================================");
    Serial.println("ESP32-CAM Web Server");
    Serial.println("================================\n");
    
    // Flash LED pini
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, LOW);
    
    // Kamera başlat
    if (!initCamera()) {
        Serial.println("Kamera hatasi!");
        while(1) delay(1000);
    }
    Serial.println("Kamera hazir!");
    
    // WiFi bağlan
    Serial.printf("WiFi baglaniyor: %s\n", ssid);
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n\nWiFi baglandi!");
        Serial.printf("IP Adresi: http://%s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nWiFi baglantisi basarisiz!");
        Serial.println("Access Point moduna geciliyor...");
        
        // AP modu
        WiFi.softAP("ESP32-CAM", "12345678");
        Serial.printf("AP IP: http://%s\n", WiFi.softAPIP().toString().c_str());
    }
    
    // Web server route'lar
    server.on("/", handleRoot);
    server.on("/capture", handleCapture);
    server.on("/stream", handleStream);
    
    server.begin();
    Serial.println("Web server baslatildi!");
    Serial.println("Tarayicinizda IP adresini acin.");
}

// ==================== LOOP ====================
void loop() {
    server.handleClient();
    delay(1);
}
