/*
 * ===================================================
 * EE4065 Final Project - Question 2
 * ALTERNATIF: Basit CNN ile Digit Detection
 * ===================================================
 * 
 * Bu kod, YOLO yerine basit bir CNN sınıflandırıcı kullanır.
 * Model boyutu çok daha küçük (~50-100KB) olacağı için
 * ESP32-CAM'de daha rahat çalışır.
 * 
 * YOLO modeli çok büyük olursa bu alternatifi kullanın.
 * 
 * Avantajları:
 * - Çok küçük model boyutu (~50KB)
 * - Hızlı inference (~50ms)
 * - Daha az bellek kullanımı
 * 
 * Dezavantajları:
 * - Sadece sınıflandırma (detection değil)
 * - Tek rakam tespiti (görüntüde tek rakam olmalı)
 * 
 * ===================================================
 */

#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <WiFi.h>
#include "esp_http_server.h"

// Basit model için include
#include "simple_digit_model.h"

// ===================================================
// Konfigürasyon
// ===================================================

// WiFi bilgileri
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Model parametreleri
#define INPUT_SIZE 28      // MNIST boyutu
#define NUM_CLASSES 10

// ===================================================
// Camera Pin Tanımları (AI-Thinker ESP32-CAM)
// ===================================================
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
#define LED_GPIO_NUM       4

// ===================================================
// Global Değişkenler
// ===================================================
const char* digit_labels[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
httpd_handle_t camera_httpd = NULL;

// Ağ ağırlıkları (simple_digit_model.h'dan gelecek)
extern const float conv1_weights[];
extern const float conv1_bias[];
extern const float conv2_weights[];
extern const float conv2_bias[];
extern const float fc1_weights[];
extern const float fc1_bias[];
extern const float fc2_weights[];
extern const float fc2_bias[];

// ===================================================
// Aktivasyon Fonksiyonları
// ===================================================
inline float relu(float x) {
    return x > 0 ? x : 0;
}

void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }
    
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// ===================================================
// Basit CNN Forward Pass
// ===================================================
// Model mimarisi:
// Input (28x28x1) -> Conv1 (3x3, 8 filters) -> ReLU -> MaxPool (2x2)
// -> Conv2 (3x3, 16 filters) -> ReLU -> MaxPool (2x2)
// -> Flatten -> FC1 (128) -> ReLU -> FC2 (10) -> Softmax

class SimpleCNN {
private:
    // Ara sonuçlar için buffer'lar
    float conv1_out[26 * 26 * 8];    // 28-3+1 = 26
    float pool1_out[13 * 13 * 8];    // 26/2 = 13
    float conv2_out[11 * 11 * 16];   // 13-3+1 = 11
    float pool2_out[5 * 5 * 16];     // 11/2 = 5 (floor)
    float fc1_out[128];
    float fc2_out[10];
    
public:
    int predict(float* input_28x28, float* confidence) {
        // Conv1: 28x28x1 -> 26x26x8
        conv2d(input_28x28, 28, 28, 1, conv1_weights, conv1_bias, conv1_out, 26, 26, 8, 3);
        applyRelu(conv1_out, 26 * 26 * 8);
        
        // MaxPool1: 26x26x8 -> 13x13x8
        maxpool2d(conv1_out, 26, 26, 8, pool1_out, 13, 13);
        
        // Conv2: 13x13x8 -> 11x11x16
        conv2d(pool1_out, 13, 13, 8, conv2_weights, conv2_bias, conv2_out, 11, 11, 16, 3);
        applyRelu(conv2_out, 11 * 11 * 16);
        
        // MaxPool2: 11x11x16 -> 5x5x16
        maxpool2d(conv2_out, 11, 11, 16, pool2_out, 5, 5);
        
        // Flatten + FC1: 5*5*16=400 -> 128
        fullyConnected(pool2_out, 400, fc1_weights, fc1_bias, fc1_out, 128);
        applyRelu(fc1_out, 128);
        
        // FC2: 128 -> 10
        fullyConnected(fc1_out, 128, fc2_weights, fc2_bias, fc2_out, 10);
        
        // Softmax
        softmax(fc2_out, 10);
        
        // En yüksek skoru bul
        int max_idx = 0;
        float max_val = fc2_out[0];
        for (int i = 1; i < 10; i++) {
            if (fc2_out[i] > max_val) {
                max_val = fc2_out[i];
                max_idx = i;
            }
        }
        
        *confidence = max_val;
        return max_idx;
    }
    
private:
    void conv2d(float* input, int in_h, int in_w, int in_c,
                const float* weights, const float* bias,
                float* output, int out_h, int out_w, int out_c, int kernel_size) {
        int k = kernel_size;
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh + kh;
                                int iw = ow + kw;
                                int input_idx = (ih * in_w + iw) * in_c + ic;
                                int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                    output[(oh * out_w + ow) * out_c + oc] = sum;
                }
            }
        }
    }
    
    void maxpool2d(float* input, int in_h, int in_w, int channels,
                   float* output, int out_h, int out_w) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -1e9;
                    for (int ph = 0; ph < 2; ph++) {
                        for (int pw = 0; pw < 2; pw++) {
                            int ih = oh * 2 + ph;
                            int iw = ow * 2 + pw;
                            if (ih < in_h && iw < in_w) {
                                int idx = (ih * in_w + iw) * channels + c;
                                if (input[idx] > max_val) {
                                    max_val = input[idx];
                                }
                            }
                        }
                    }
                    output[(oh * out_w + ow) * channels + c] = max_val;
                }
            }
        }
    }
    
    void fullyConnected(float* input, int in_size,
                        const float* weights, const float* bias,
                        float* output, int out_size) {
        for (int o = 0; o < out_size; o++) {
            float sum = bias[o];
            for (int i = 0; i < in_size; i++) {
                sum += input[i] * weights[o * in_size + i];
            }
            output[o] = sum;
        }
    }
    
    void applyRelu(float* data, int size) {
        for (int i = 0; i < size; i++) {
            data[i] = relu(data[i]);
        }
    }
};

SimpleCNN cnn;

// ===================================================
// Görüntü Ön İşleme
// ===================================================
void preprocessTo28x28(camera_fb_t* fb, float* output) {
    // RGB565'ten 28x28 grayscale'e dönüştür
    // Bilinear interpolation ile resize
    
    uint16_t* img = (uint16_t*)fb->buf;
    int src_w = fb->width;
    int src_h = fb->height;
    
    float scale_x = (float)src_w / INPUT_SIZE;
    float scale_y = (float)src_h / INPUT_SIZE;
    
    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            // Kaynak koordinatları
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            
            // Sınır kontrolü
            if (x0 >= src_w - 1) x0 = src_w - 2;
            if (y0 >= src_h - 1) y0 = src_h - 2;
            
            // RGB565'ten grayscale
            uint16_t pixel = img[y0 * src_w + x0];
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5) & 0x3F) << 2;
            uint8_t b = (pixel & 0x1F) << 3;
            
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            
            // Normalize (0-1 arası)
            output[y * INPUT_SIZE + x] = gray / 255.0f;
        }
    }
    
    // İnvert (siyah arka plan, beyaz rakam için)
    // MNIST formatına uygun hale getir
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        output[i] = 1.0f - output[i];
    }
}

// ===================================================
// Kamera Başlatma
// ===================================================
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
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_QVGA;  // 320x240
    config.jpeg_quality = 12;
    config.fb_count = 1;
    
    if (psramFound()) {
        config.fb_count = 2;
        config.grab_mode = CAMERA_GRAB_LATEST;
    }
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Kamera başlatılamadı: 0x%x\n", err);
        return false;
    }
    
    Serial.println("Kamera başlatıldı!");
    return true;
}

// ===================================================
// Digit Detection
// ===================================================
int detectDigit(camera_fb_t* fb, float* confidence) {
    float input[INPUT_SIZE * INPUT_SIZE];
    
    // Görüntüyü 28x28'e dönüştür
    preprocessTo28x28(fb, input);
    
    // CNN ile tahmin
    unsigned long start = millis();
    int digit = cnn.predict(input, confidence);
    unsigned long elapsed = millis() - start;
    
    Serial.printf("Inference: %lu ms\n", elapsed);
    
    return digit;
}

// ===================================================
// HTTP Handlers
// ===================================================
static esp_err_t index_handler(httpd_req_t *req) {
    const char* html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESP32 Digit Detection (Simple)</title>
    <style>
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            text-align: center;
            padding: 20px;
        }
        h1 { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
        .result {
            font-size: 10em;
            font-weight: bold;
            color: #00ff00;
            text-shadow: 0 0 30px #00ff00;
        }
        .conf { font-size: 2em; color: #888; }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 20px 40px;
            font-size: 1.5em;
            cursor: pointer;
            margin: 20px;
        }
        button:hover { background: #00cc00; }
        img { max-width: 320px; border: 2px solid #00ff00; }
    </style>
</head>
<body>
    <h1>🔢 Simple Digit Detection</h1>
    <p>Basit CNN Model (MNIST tarzı)</p>
    <img id="capture" src="">
    <div class="result" id="digit">-</div>
    <div class="conf" id="conf">Güven: --%</div>
    <button onclick="detect()">📸 Tespit Et</button>
    
    <script>
        async function detect() {
            try {
                // Capture image
                const imgRes = await fetch('/capture');
                const blob = await imgRes.blob();
                document.getElementById('capture').src = URL.createObjectURL(blob);
                
                // Get detection result
                const res = await fetch('/detect');
                const data = await res.json();
                
                document.getElementById('digit').textContent = data.digit;
                document.getElementById('conf').textContent = 
                    'Güven: ' + (data.confidence * 100).toFixed(1) + '%';
            } catch(e) {
                console.error(e);
            }
        }
    </script>
</body>
</html>
)rawliteral";
    
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, strlen(html));
}

static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // JPEG'e dönüştür
    size_t jpg_len;
    uint8_t *jpg_buf;
    bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
    esp_camera_fb_return(fb);
    
    if (!converted) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_send(req, (const char*)jpg_buf, jpg_len);
    free(jpg_buf);
    
    return ESP_OK;
}

static esp_err_t detect_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    float confidence;
    int digit = detectDigit(fb, &confidence);
    esp_camera_fb_return(fb);
    
    char json[64];
    snprintf(json, sizeof(json), 
             "{\"digit\":\"%s\",\"confidence\":%.4f}",
             digit_labels[digit], confidence);
    
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, json, strlen(json));
}

void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    
    httpd_uri_t index_uri = { "/", HTTP_GET, index_handler, NULL };
    httpd_uri_t capture_uri = { "/capture", HTTP_GET, capture_handler, NULL };
    httpd_uri_t detect_uri = { "/detect", HTTP_GET, detect_handler, NULL };
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        httpd_register_uri_handler(camera_httpd, &detect_uri);
        Serial.println("HTTP Server başlatıldı!");
    }
}

// ===================================================
// Setup & Loop
// ===================================================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    Serial.println("\n=============================");
    Serial.println("Simple Digit Detection");
    Serial.println("EE4065 - Question 2 (Alt.)");
    Serial.println("=============================\n");
    
    pinMode(LED_GPIO_NUM, OUTPUT);
    
    if (!initCamera()) {
        Serial.println("Kamera hatası!");
        while(1) delay(1000);
    }
    
    // WiFi bağlantısı
    WiFi.begin(ssid, password);
    Serial.print("WiFi bağlanıyor");
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("\nWiFi Bağlandı!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    
    startServer();
    
    Serial.println("\nHazır! Tarayıcıdan IP adresine gidin.");
}

void loop() {
    // Seri komutları
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'd' || c == 'D') {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) {
                float conf;
                int digit = detectDigit(fb, &conf);
                Serial.printf("\nSonuç: %s (%.1f%%)\n", 
                             digit_labels[digit], conf * 100);
                esp_camera_fb_return(fb);
            }
        }
    }
    delay(10);
}
