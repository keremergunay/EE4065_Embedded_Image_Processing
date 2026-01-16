/*
 * EE4065 Final Project - Question 2
 * Handwritten Digit Detection on ESP32-CAM
 * 
 * RHYX M21-45 (GC2145) kamera sensörü ile uyumlu
 * TensorFlow Lite Micro + YOLO tabanlı tespit
 */

#include "esp_camera.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <WiFi.h>
#include "esp_http_server.h"

// TensorFlow Lite
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Custom YOLO-Nano Model
// >>> MODEL SEÇİMİ: Hangisini kullanmak istiyorsan uncomment yap <<<
#define USE_ROBOFLOW_MODEL false  // true = Roboflow, false = MNIST

#if USE_ROBOFLOW_MODEL
    #include "yolo_model_roboflow.h"
    #define yolo_input_scale yolo_roboflow_input_scale
    #define yolo_input_zero_point yolo_roboflow_input_zero_point
    #define yolo_model yolo_roboflow_model
    #define yolo_model_len yolo_roboflow_model_len
#else
    #include "yolo_model_mnist.h"
    #define yolo_input_scale yolo_mnist_input_scale
    #define yolo_input_zero_point yolo_mnist_input_zero_point
    #define yolo_model yolo_mnist_model
    #define yolo_model_len yolo_mnist_model_len
#endif

// ==================== WIFI AYARLARI ====================
// 
// >>> KULLANIM: Aşağıdaki satırı değiştir <<<
//   true  = AP Mode (Okul - portlar kapalıyken)
//   false = Station Mode (Ev - mevcut WiFi'ye bağlan)
//
#define USE_AP_MODE false

// ----- AP Mode (ESP32 kendi WiFi ağını oluşturur) -----
// Bağlantı: ESP32'nin oluşturduğu ağa bağlan, IP: 192.168.4.1
const char* ap_ssid = "ESP32-CAM-Digit";
const char* ap_password = "12345678";

// ----- Station Mode (Ev WiFi'sine bağlan) -----
// Bağlantı: Serial Monitor'den IP adresini öğren
const char* sta_ssid = "iPhone SE";
const char* sta_password = "404404404";

#define INPUT_WIDTH  96
#define INPUT_HEIGHT 96
#define GRID_SIZE 6
#define NUM_ANCHORS 2
#define NUM_CLASSES 10

// Tensor arena - model 122KB, arena ~200KB yeterli
constexpr int kTensorArenaSize = 200 * 1024;
uint8_t* tensor_arena = nullptr;

// ==================== PIN TANIMLARI (AI-Thinker) ====================
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

// ==================== GLOBAL DEĞİŞKENLER ====================
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

struct Detection {
    int class_id;
    float confidence;
    float x, y, w, h;
};

#define MAX_DETECTIONS 10
Detection detections[MAX_DETECTIONS];
int num_detections = 0;

// Class labels for digits 0-9
const char* digit_labels[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

// Son çekilen fotoğraf (snapshot için)
uint8_t* last_snapshot = nullptr;
size_t last_snapshot_len = 0;
SemaphoreHandle_t snapshot_mutex = nullptr;

// ==================== THRESHOLD AYARLARI ====================
int g_threshold = 128;       // 0-255 arası threshold değeri
bool g_threshold_enabled = false;  // Threshold aktif mi?
bool g_invert = false;       // Tersine çevir (beyaz arka plan için)

// ==================== KAMERA ====================
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
    
    // GC2145 için ayarlar - GRAYSCALE format
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    
    if(psramFound()) {
        Serial.println("PSRAM bulundu!");
        config.frame_size = FRAMESIZE_QVGA;  // 320x240, sonra 96x96'ya resize edilecek
        config.jpeg_quality = 12;
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        Serial.println("PSRAM bulunamadı!");
        config.frame_size = FRAMESIZE_QQVGA;  // 160x120, sonra 96x96'ya resize edilecek
        config.jpeg_quality = 15;
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    }
    
    // Kamerayı aktif et
    if(PWDN_GPIO_NUM != -1) {
        pinMode(PWDN_GPIO_NUM, OUTPUT);
        digitalWrite(PWDN_GPIO_NUM, LOW);
        delay(10);
    }
    
    delay(100);
    Serial.println("Kamera başlatılıyor (Custom YOLO-Nano)...");
    esp_err_t err = esp_camera_init(&config);
    
    // Başarısız olursa düşük ayarlarla tekrar dene
    if (err != ESP_OK) {
        Serial.printf("İlk deneme başarısız: 0x%x, tekrar deneniyor...\n", err);
        esp_camera_deinit();
        delay(100);
        
        config.xclk_freq_hz = 8000000;
        config.frame_size = FRAMESIZE_QQVGA;  // 160x120
        err = esp_camera_init(&config);
        
        if (err != ESP_OK) {
            Serial.printf("Kamera başlatılamadı! Hata: 0x%x\n", err);
            return false;
        }
    }
    
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        Serial.printf("Kamera PID: 0x%x\n", s->id.PID);
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
    }
    
    Serial.println("Kamera hazır!");
    return true;
}

// ==================== TENSORFLOW LITE ====================
bool initTFLite() {
    Serial.println("TensorFlow Lite başlatılıyor...");
    
    tensor_arena = psramFound() ? (uint8_t*)ps_malloc(kTensorArenaSize) 
                                 : (uint8_t*)malloc(kTensorArenaSize);
    
    if (!tensor_arena) {
        Serial.println("Tensor arena oluşturulamadı!");
        return false;
    }
    
    static tflite::MicroErrorReporter error_reporter;
    const tflite::Model* model = tflite::GetModel(yolo_model);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model sürümü uyumsuz!");
        return false;
    }
    
    // Sadece gerekli operatörleri ekle (hızlandırma için)
    static tflite::MicroMutableOpResolver<25> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddLogistic();  // Sigmoid
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMean();      // BatchNorm için
    resolver.AddRsqrt();     // BatchNorm için
    resolver.AddSub();       // BatchNorm için
    resolver.AddConcatenation();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddPad();
    resolver.AddQuantize();      // INT8 quantization
    resolver.AddDequantize();    // INT8 dequantization
    resolver.AddLeakyRelu();     // LeakyReLU (Custom YOLO-Nano için)
    resolver.AddShape();         // Shape operatörü
    resolver.AddGather();        // Gather operatörü
    resolver.AddExpandDims();    // ExpandDims operatörü
    resolver.AddFill();          // Fill operatörü
    
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensör tahsisi başarısız!");
        return false;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Input bilgisi
    Serial.printf("Input dims: %d -> [", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        Serial.printf("%d", input->dims->data[i]);
        if (i < input->dims->size - 1) Serial.print(",");
    }
    Serial.println("]");
    
    // Output bilgisi
    Serial.printf("Output dims: %d -> [", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        Serial.printf("%d", output->dims->data[i]);
        if (i < output->dims->size - 1) Serial.print(",");
    }
    Serial.println("]");
    
    // Output scale/zp
    Serial.printf("Output scale: %.6f, zero_point: %d\n", 
                  output->params.scale, output->params.zero_point);
    
    Serial.println("TensorFlow Lite (YOLO-Nano) hazır!");
    return true;
}

// ==================== GÖRÜNTÜ İŞLEME ====================
// Kamera görüntüsünü 96x96'ya resize et (GRAYSCALE format)
// Q4'ten alınan: Contrast Stretch eklendi (preprocessing iyileştirmesi)
void preprocessImage(camera_fb_t* fb, int8_t* data) {
    uint8_t* img = fb->buf;  // Grayscale: 1 byte per pixel
    int src_w = fb->width;
    int src_h = fb->height;
    
    // Temporary buffer for grayscale values (for contrast stretch)
    static uint8_t gray_temp[INPUT_WIDTH * INPUT_HEIGHT];
    
    // Bilinear interpolation ile resize
    float x_ratio = (float)src_w / INPUT_WIDTH;
    float y_ratio = (float)src_h / INPUT_HEIGHT;
    
    uint8_t min_val = 255, max_val = 0;
    
    for (int y = 0; y < INPUT_HEIGHT; y++) {
        for (int x = 0; x < INPUT_WIDTH; x++) {
            // Kaynak koordinatları
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = min(x0 + 1, src_w - 1);
            int y1 = min(y0 + 1, src_h - 1);
            
            float x_diff = src_x - x0;
            float y_diff = src_y - y0;
            
            // 4 komşu piksel (grayscale - doğrudan uint8_t)
            uint8_t g00 = img[y0 * src_w + x0];
            uint8_t g01 = img[y0 * src_w + x1];
            uint8_t g10 = img[y1 * src_w + x0];
            uint8_t g11 = img[y1 * src_w + x1];
            
            // Bilinear interpolation
            float gray = g00 * (1 - x_diff) * (1 - y_diff) +
                        g01 * x_diff * (1 - y_diff) +
                        g10 * (1 - x_diff) * y_diff +
                        g11 * x_diff * y_diff;
            
            uint8_t gray_val = constrain((int)gray, 0, 255);
            gray_temp[y * INPUT_WIDTH + x] = gray_val;
            
            // Track min/max for contrast stretch
            if (gray_val < min_val) min_val = gray_val;
            if (gray_val > max_val) max_val = gray_val;
        }
    }
    
    // --- CONTRAST STRETCH (Q4'ten alındı) ---
    // Histogram'ı [0-255] aralığına yay
    int range = max_val - min_val;
    if (range < 10) range = 10;  // Çok düşük kontrast durumunda sıfıra bölmeyi önle
    
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; i++) {
        int stretched = ((gray_temp[i] - min_val) * 255) / range;
        stretched = constrain(stretched, 0, 255);
        
        // --- BINARY THRESHOLDING (Web'den ayarlanabilir) ---
        if (g_threshold_enabled) {
            // İnvert: Beyaz kağıt üzerine siyah yazı için
            if (g_invert) {
                stretched = (stretched < g_threshold) ? 255 : 0;
            } else {
                stretched = (stretched >= g_threshold) ? 255 : 0;
            }
        }
        
        // INT8 quantization using model's input parameters
        // yolo_input_scale = 0.00378, yolo_input_zero_point = -128
        // Formula: quantized = float_value / scale + zero_point
        // For normalized [0,1] input: q = (stretched/255.0) / scale + zero_point
        // Simplified: q = stretched / (255 * scale) + zero_point
        int8_t val = (int8_t)((stretched / 255.0f) / yolo_input_scale + yolo_input_zero_point);
        
        // Single channel output (model expects grayscale)
        data[i] = val;
    }
}

float calculateIoU(Detection& a, Detection& b) {
    float x1 = max(a.x - a.w/2, b.x - b.w/2);
    float y1 = max(a.y - a.h/2, b.y - b.h/2);
    float x2 = min(a.x + a.w/2, b.x + b.w/2);
    float y2 = min(a.y + a.h/2, b.y + b.h/2);
    
    float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float union_area = a.w * a.h + b.w * b.h - intersection;
    return intersection / (union_area + 1e-6);
}

// ==================== Custom YOLO-Nano DECODING ====================
int detectDigits(camera_fb_t* fb) {
    num_detections = 0;
    if (!fb || !fb->buf) return 0;
    
    preprocessImage(fb, input->data.int8);
    
    unsigned long t = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference basarisiz!");
        return 0;
    }
    long inference_time = millis() - t;
    Serial.printf("Inference: %lu ms\n", inference_time);
    
    // Output type check
    if (output->type == kTfLiteFloat32) {
        Serial.println("Output type: Float32 (Correct)");
    } else if (output->type == kTfLiteInt8) {
        Serial.println("Output type: Int8 (Unexpected!)");
    }

    float* out = output->data.f; // Use float pointer directly
    // float scale = output->params.scale; // Not needed for float32
    // int zp = output->params.zero_point; // Not needed for float32
    
    // DEBUG: Print first 20 values
    Serial.print("First 20 out: ");
    for (int i = 0; i < 20; i++) {
        Serial.printf("%.2f ", out[i]);
    }
    Serial.println();
    
    float conf_thresh = 0.05; // Çok düşük - debug için
    float nms_thresh = 0.30;
    
    Detection temp[MAX_DETECTIONS * 3]; 
    int temp_count = 0;
    
    // Output Shape: [1, 6, 6, 2, 15] (Batch, GridY, GridX, Anchor, Values)
    // Flattened index mapping needed
    int values_per_anchor = 5 + NUM_CLASSES; // 15
    int anchors_per_cell = NUM_ANCHORS;      // 2
    int total_cells = GRID_SIZE * GRID_SIZE; // 36
    
    // DEBUG: Find max objectness across all cells
    float max_obj = -1000;
    int max_gy = 0, max_gx = 0, max_a = 0;
    
    for (int gy = 0; gy < GRID_SIZE; gy++) {
        for (int gx = 0; gx < GRID_SIZE; gx++) {
            for (int a = 0; a < anchors_per_cell; a++) {
                // Calculate index in flattened array
                // Index formula depends on TFLite export (usually NHWC)
                // Assuming [GridY, GridX, Anchor, Values]
                int cell_idx = ((gy * GRID_SIZE + gx) * anchors_per_cell + a) * values_per_anchor;
                
                // Helper: Sigmoid function
                auto sigmoid = [](float x) -> float {
                    return 1.0f / (1.0f + expf(-x));
                };
                
                // 1. Objectness (Index 0) - Apply Sigmoid!
                float obj_raw = out[cell_idx + 0];
                float obj = sigmoid(obj_raw);
                
                // Track max objectness
                if (obj > max_obj) {
                    max_obj = obj;
                    max_gy = gy;
                    max_gx = gx;
                    max_a = a;
                }
                
                if (obj < conf_thresh) continue;
                
                // 4. Classes (Index 5-14) - Apply Softmax
                int best_cls = 0;
                float best_prob = 0;
                float class_sum = 0;
                float class_probs[NUM_CLASSES];
                
                // Softmax: exp(x_i) / sum(exp(x_j))
                for (int c = 0; c < NUM_CLASSES; c++) {
                    class_probs[c] = expf(out[cell_idx + 5 + c]);
                    class_sum += class_probs[c];
                }
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float prob = class_probs[c] / class_sum;
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_cls = c;
                    }
                }
                
                float score = obj * best_prob;
                if (score < conf_thresh) continue;
                
                // 2. Coordinates (Index 1-4) - Apply Sigmoid!
                float bx = sigmoid(out[cell_idx + 1]); // Center X relative to cell (0-1)
                float by = sigmoid(out[cell_idx + 2]); // Center Y relative to cell (0-1)
                float bw = sigmoid(out[cell_idx + 3]); // Width (0-1)
                float bh = sigmoid(out[cell_idx + 4]); // Height (0-1)
                
                // YOLO-Nano Decoding (Direct Regression)
                float x_center = (gx + bx) / GRID_SIZE;
                float y_center = (gy + by) / GRID_SIZE;
                
                if (temp_count < MAX_DETECTIONS * 3) {
                    temp[temp_count].class_id = best_cls;
                    temp[temp_count].confidence = score;
                    temp[temp_count].x = x_center;
                    temp[temp_count].y = y_center;
                    temp[temp_count].w = bw;
                    temp[temp_count].h = bh;
                    temp_count++;
                }
            }
        }
    }
    
    // DEBUG: Print max objectness found
    Serial.printf("Max objectness: %.4f at cell(%d,%d) anchor=%d\n", max_obj, max_gx, max_gy, max_a);
    Serial.printf("Candidates before NMS: %d\n", temp_count);
    
    // NMS
    bool suppressed[60] = {false}; // temp_count limit
    for (int i = 0; i < temp_count; i++) {
        if (suppressed[i]) continue;
        
        // Add to final detections
        if (num_detections < MAX_DETECTIONS) {
            detections[num_detections++] = temp[i];
            
            // Suppress IoU > thresh
            for (int j = i + 1; j < temp_count; j++) {
                if (!suppressed[j] && calculateIoU(temp[i], temp[j]) > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    Serial.printf("Detected: %d digits\n", num_detections);
    return num_detections;
}

void setFlash(bool on) { digitalWrite(LED_GPIO_NUM, on ? HIGH : LOW); }

// ==================== WEB ARAYUZU ====================
static esp_err_t index_handler(httpd_req_t *req) {
    const char* html = 
"<!DOCTYPE html>"
"<html><head>"
"<meta charset=\"UTF-8\">"
"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1.0\">"
"<title>Digit Detection</title>"
"<style>"
"*{margin:0;padding:0;box-sizing:border-box}"
"body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#0f0f23,#1a1a3e);color:#eee;min-height:100vh;padding:20px}"
"h1{text-align:center;font-size:2.2em;margin-bottom:5px;background:linear-gradient(90deg,#4ecca3,#7dd3fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}"
".sub{text-align:center;color:#666;margin-bottom:30px}"
".main{display:flex;flex-wrap:wrap;justify-content:center;gap:25px;max-width:1100px;margin:0 auto}"
".panel{background:rgba(255,255,255,0.05);border-radius:16px;padding:25px;border:1px solid rgba(255,255,255,0.1)}"
".photo-panel{flex:2;min-width:400px;max-width:600px}"
".results-panel{flex:1;min-width:280px;max-width:380px}"
".panel h2{font-size:1.1em;color:#7dd3fc;margin-bottom:15px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.1)}"
"#canvas{width:100%;border-radius:12px;background:#111;display:block}"
".placeholder{text-align:center;padding:80px 20px;color:#555;background:#111;border-radius:12px}"
".placeholder p{font-size:4em;margin-bottom:15px}"
".btns{display:flex;flex-wrap:wrap;gap:10px;margin-top:20px;justify-content:center}"
"button{background:linear-gradient(135deg,#4ecca3,#38b593);color:#0f0f23;border:none;padding:14px 24px;font-size:1em;font-weight:600;border-radius:12px;cursor:pointer;transition:all 0.2s}"
"button:hover{transform:translateY(-2px);box-shadow:0 5px 25px rgba(78,204,163,0.4)}"
"button:active{transform:translateY(0)}"
".btn-capture{background:linear-gradient(135deg,#f472b6,#db2777);font-size:1.15em;padding:16px 32px}"
".btn-flash{background:linear-gradient(135deg,#fbbf24,#f59e0b)}"
".btn-save{background:linear-gradient(135deg,#60a5fa,#3b82f6)}"
".btn-new{background:linear-gradient(135deg,#a78bfa,#7c3aed)}"
"#status{text-align:center;font-size:1.1em;color:#fbbf24;margin:15px 0;min-height:28px}"
"#stats{text-align:center;font-size:0.85em;color:#666;margin-top:10px}"
"#list{max-height:400px;overflow-y:auto}"
".det-item{background:rgba(78,204,163,0.15);border-left:4px solid #4ecca3;padding:14px;margin:12px 0;border-radius:0 12px 12px 0}"
".det-digit{font-size:2.5em;font-weight:bold;color:#4ecca3}"
".det-conf{color:#94a3b8;font-size:0.9em;margin-top:5px}"
".det-bbox{font-family:monospace;font-size:0.8em;color:#7dd3fc;background:rgba(0,0,0,0.3);padding:8px 12px;border-radius:8px;margin-top:10px}"
".no-det{color:#555;text-align:center;padding:40px 20px}"
".flash-on{box-shadow:0 0 20px #fbbf24}"
".thresh-panel{margin-top:20px;padding:15px;background:rgba(0,0,0,0.2);border-radius:12px}"
".thresh-panel h3{font-size:0.95em;color:#a78bfa;margin-bottom:12px}"
".thresh-row{display:flex;align-items:center;gap:12px;margin:10px 0}"
".thresh-row label{min-width:80px;color:#94a3b8;font-size:0.9em}"
".thresh-row input[type=range]{flex:1;height:8px;-webkit-appearance:none;background:#333;border-radius:4px;outline:none}"
".thresh-row input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;background:#a78bfa;border-radius:50%;cursor:pointer}"
".thresh-row input[type=checkbox]{width:18px;height:18px;accent-color:#a78bfa}"
".thresh-val{min-width:40px;text-align:center;color:#a78bfa;font-weight:bold}"
"</style></head><body>"
"<h1>Digit Detection</h1>"
"<p class=\"sub\">EE4065 Final Project - YOLO Tabanli Rakam Tespiti</p>"
"<div class=\"main\">"
"<div class=\"panel photo-panel\">"
"<h2>Cekilen Fotograf</h2>"
"<div id=\"photoArea\">"
"<div class=\"placeholder\" id=\"placeholder\">"
"<p>?</p>"
"<span>Fotograf cekmek icin asagidaki butona basin</span>"
"</div>"
"<canvas id=\"canvas\" style=\"display:none\"></canvas>"
"</div>"
"<div id=\"stats\"></div>"
"<div class=\"btns\">"
"<button class=\"btn-capture\" id=\"captureBtn\" onclick=\"capture()\">Fotograf Cek ve Tespit Et</button>"
"<button class=\"btn-flash\" id=\"flashBtn\" onclick=\"toggleFlash()\">Flash</button>"
"</div>"
"<div class=\"btns\" id=\"actionBtns\" style=\"display:none\">"
"<button class=\"btn-save\" onclick=\"saveImage()\">PC'ye Kaydet</button>"
"<button class=\"btn-new\" onclick=\"resetCapture()\">Yeni Fotograf</button>"
"</div>"
"</div>"
"<div class=\"panel results-panel\">"
"<h2>Tespit Sonuclari</h2>"
"<div id=\"status\">Fotograf cekilmedi</div>"
"<div id=\"list\">"
"<div class=\"no-det\">Henuz tespit yapilmadi</div>"
"</div>"
"<div class=\"thresh-panel\">"
"<h3>Threshold Ayarlari</h3>"
"<div class=\"thresh-row\">"
"<label>Aktif:</label>"
"<input type=\"checkbox\" id=\"threshEn\" onchange=\"updateThreshold()\">"
"</div>"
"<div class=\"thresh-row\">"
"<label>Deger:</label>"
"<input type=\"range\" id=\"threshVal\" min=\"0\" max=\"255\" value=\"128\" oninput=\"updateThreshold()\">"
"<span class=\"thresh-val\" id=\"threshDisp\">128</span>"
"</div>"
"<div class=\"thresh-row\">"
"<label>Invert:</label>"
"<input type=\"checkbox\" id=\"threshInv\" onchange=\"updateThreshold()\">"
"</div>"
"<div class=\"thresh-row\">"
"<button onclick=\"refreshDebug()\" style=\"padding:8px 16px;font-size:0.85em\">Debug Goruntu</button>"
"</div>"
"<img id=\"debugImg\" style=\"width:96px;height:96px;border:2px solid #a78bfa;border-radius:8px;margin-top:10px;display:none;image-rendering:pixelated\">"
"</div>"
"</div>"
"</div>"
"<script>"
"var flash=false,capturedImg=null,lastDetections=[];"
"function capture(){"
"var btn=document.getElementById('captureBtn');"
"btn.disabled=true;btn.innerText='Cekiliyor...';"
"document.getElementById('status').innerText='Fotograf cekiliyor...';"
"fetch('/capture').then(function(r){return r.json();}).then(function(data){"
"if(data.error){"
"document.getElementById('status').innerText='Hata: '+data.error;"
"btn.disabled=false;btn.innerText='Fotograf Cek ve Tespit Et';"
"return;}"
"lastDetections=data.detections||[];"
"document.getElementById('status').innerText='Goruntu yukleniyor...';"
"var img=new Image();"
"img.onload=function(){"
"capturedImg=img;"
"drawDetections();"
"document.getElementById('placeholder').style.display='none';"
"document.getElementById('canvas').style.display='block';"
"document.getElementById('actionBtns').style.display='flex';"
"document.getElementById('captureBtn').style.display='none';"
"document.getElementById('stats').innerText='Boyut: '+img.width+'x'+img.height+' | Tespit: '+lastDetections.length+' rakam';"
"updateResults();"
"};"
"img.onerror=function(){"
"document.getElementById('status').innerText='Goruntu yuklenemedi';"
"btn.disabled=false;btn.innerText='Fotograf Cek ve Tespit Et';"
"};"
"img.src='/snapshot?t='+Date.now();"
"}).catch(function(e){"
"document.getElementById('status').innerText='Baglanti hatasi';"
"btn.disabled=false;btn.innerText='Fotograf Cek ve Tespit Et';"
"});"
"}"
"function drawDetections(){"
"if(!capturedImg)return;"
"var c=document.getElementById('canvas'),ctx=c.getContext('2d');"
"c.width=capturedImg.width;c.height=capturedImg.height;"
"ctx.drawImage(capturedImg,0,0);"
"var colors=['#f472b6','#4ecca3','#60a5fa','#fbbf24','#a78bfa','#fb7185','#34d399','#38bdf8','#facc15','#c084fc'];"
"for(var i=0;i<lastDetections.length;i++){"
"var d=lastDetections[i];"
"var cx=d.x*c.width,cy=d.y*c.height,bw=d.w*c.width,bh=d.h*c.height;"
"var x1=cx-bw/2,y1=cy-bh/2;"
"var col=colors[parseInt(d.digit)%10];"
"ctx.strokeStyle=col;ctx.lineWidth=3;"
"ctx.strokeRect(x1,y1,bw,bh);"
"ctx.fillStyle=col;ctx.font='bold 20px Arial';"
"var txt=d.digit+' '+(d.confidence*100).toFixed(0)+'%';"
"var tw=ctx.measureText(txt).width+10;"
"ctx.fillRect(x1,y1-25,tw,25);"
"ctx.fillStyle='#000';ctx.fillText(txt,x1+5,y1-7);"
"}"
"}"
"function updateResults(){"
"var list=document.getElementById('list');"
"if(lastDetections.length===0){"
"document.getElementById('status').innerText='Rakam bulunamadi';"
"list.innerHTML='<div class=\"no-det\">Kameraya rakam gosterin</div>';"
"return;"
"}"
"document.getElementById('status').innerText=lastDetections.length+' rakam tespit edildi';"
"var h='';"
"for(var i=0;i<lastDetections.length;i++){"
"var d=lastDetections[i];"
"var px=Math.round(d.x*160),py=Math.round(d.y*120),pw=Math.round(d.w*160),ph=Math.round(d.h*120);"
"h+='<div class=\"det-item\">';"
"h+='<span class=\"det-digit\">'+d.digit+'</span>';"
"h+='<div class=\"det-conf\">Guven: '+(d.confidence*100).toFixed(1)+'%</div>';"
"h+='<div class=\"det-bbox\">x:'+px+' y:'+py+' w:'+pw+' h:'+ph+'</div>';"
"h+='</div>';"
"}"
"list.innerHTML=h;"
"}"
"function saveImage(){"
"if(!capturedImg)return;"
"var c=document.getElementById('canvas');"
"var link=document.createElement('a');"
"var ts=new Date().toISOString().replace(/[:.]/g,'-');"
"link.download='digit_detection_'+ts+'.png';"
"link.href=c.toDataURL('image/png');"
"link.click();"
"}"
"function resetCapture(){"
"capturedImg=null;lastDetections=[];"
"document.getElementById('placeholder').style.display='block';"
"document.getElementById('canvas').style.display='none';"
"document.getElementById('actionBtns').style.display='none';"
"document.getElementById('captureBtn').style.display='inline-block';"
"document.getElementById('captureBtn').disabled=false;"
"document.getElementById('captureBtn').innerText='Fotograf Cek ve Tespit Et';"
"document.getElementById('stats').innerText='';"
"document.getElementById('status').innerText='Fotograf cekilmedi';"
"document.getElementById('list').innerHTML='<div class=\"no-det\">Henuz tespit yapilmadi</div>';"
"}"
"function toggleFlash(){"
"flash=!flash;"
"var btn=document.getElementById('flashBtn');"
"if(flash){btn.classList.add('flash-on');btn.innerText='Flash (Acik)';}"
"else{btn.classList.remove('flash-on');btn.innerText='Flash';}"
"fetch('/flash?state='+(flash?'1':'0'));"
"}"
"function updateThreshold(){"
"var en=document.getElementById('threshEn').checked?1:0;"
"var val=document.getElementById('threshVal').value;"
"var inv=document.getElementById('threshInv').checked?1:0;"
"document.getElementById('threshDisp').innerText=val;"
"fetch('/threshold?val='+val+'&en='+en+'&inv='+inv);"
"}"
"function refreshDebug(){"
"var img=document.getElementById('debugImg');"
"img.src='/debug_input?t='+Date.now();"
"img.style.display='block';"
"}"
"</script></body></html>";
    
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, strlen(html));
}

#define BOUNDARY "123456789000000000000987654321"
static const char* STREAM_TYPE = "multipart/x-mixed-replace;boundary=" BOUNDARY;
static const char* STREAM_BOUND = "\r\n--" BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char buf[64];
    
    httpd_resp_set_type(req, STREAM_TYPE);
    
    while(true) {
        fb = esp_camera_fb_get();
        if (!fb) { res = ESP_FAIL; break; }
        
        size_t jpg_len = 0;
        uint8_t *jpg = NULL;
        bool ok = frame2jpg(fb, 80, &jpg, &jpg_len);
        esp_camera_fb_return(fb);
        
        if(!ok) { res = ESP_FAIL; break; }
        
        httpd_resp_send_chunk(req, STREAM_BOUND, strlen(STREAM_BOUND));
        size_t hlen = snprintf(buf, 64, STREAM_PART, jpg_len);
        httpd_resp_send_chunk(req, buf, hlen);
        res = httpd_resp_send_chunk(req, (const char*)jpg, jpg_len);
        free(jpg);
        
        if(res != ESP_OK) break;
        delay(30);
    }
    return res;
}

static esp_err_t detect_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }
    
    int count = detectDigits(fb);
    esp_camera_fb_return(fb);
    
    char json[1024];
    int off = snprintf(json, sizeof(json), "{\"count\":%d,\"detections\":[", count);
    
    for (int i = 0; i < count && i < MAX_DETECTIONS; i++) {
        Detection& d = detections[i];
        if (i > 0) off += snprintf(json + off, sizeof(json) - off, ",");
        off += snprintf(json + off, sizeof(json) - off,
            "{\"digit\":\"%s\",\"confidence\":%.3f,\"x\":%.3f,\"y\":%.3f,\"w\":%.3f,\"h\":%.3f}",
            digit_labels[d.class_id], d.confidence, d.x, d.y, d.w, d.h);
    }
    snprintf(json + off, sizeof(json) - off, "]}");
    
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, json, strlen(json));
}

// Capture: Fotoğraf çek, tespit yap, JPEG'i sakla
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, "{\"error\":\"Kamera hatasi\"}", -1);
        return ESP_FAIL;
    }
    
    // Tespit yap
    int count = detectDigits(fb);
    
    // RGB565'i JPEG'e dönüştür
    size_t jpg_len = 0;
    uint8_t *jpg_buf = NULL;
    bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
    esp_camera_fb_return(fb);
    
    if (!converted || !jpg_buf) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, "{\"error\":\"JPEG donusumu basarisiz\"}", -1);
        return ESP_FAIL;
    }
    
    Serial.printf("Capture: %d tespit, JPEG %d bytes\n", count, jpg_len);
    
    // Snapshot'ı sakla (thread-safe)
    if (xSemaphoreTake(snapshot_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        if (last_snapshot) free(last_snapshot);
        last_snapshot = jpg_buf;
        last_snapshot_len = jpg_len;
        xSemaphoreGive(snapshot_mutex);
    } else {
        free(jpg_buf);
    }
    
    // Sadece tespit sonuçlarını JSON olarak döndür
    char json[1024];
    int off = snprintf(json, sizeof(json), "{\"count\":%d,\"detections\":[", count);
    
    for (int i = 0; i < count && i < MAX_DETECTIONS; i++) {
        Detection& d = detections[i];
        if (i > 0) off += snprintf(json + off, sizeof(json) - off, ",");
        off += snprintf(json + off, sizeof(json) - off,
            "{\"digit\":\"%s\",\"confidence\":%.3f,\"x\":%.3f,\"y\":%.3f,\"w\":%.3f,\"h\":%.3f}",
            digit_labels[d.class_id], d.confidence, d.x, d.y, d.w, d.h);
    }
    snprintf(json + off, sizeof(json) - off, "]}");
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// Snapshot: Son çekilen fotoğrafı JPEG olarak döndür
static esp_err_t snapshot_handler(httpd_req_t *req) {
    if (xSemaphoreTake(snapshot_mutex, pdMS_TO_TICKS(500)) == pdTRUE) {
        if (last_snapshot && last_snapshot_len > 0) {
            httpd_resp_set_type(req, "image/jpeg");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
            esp_err_t res = httpd_resp_send(req, (const char*)last_snapshot, last_snapshot_len);
            xSemaphoreGive(snapshot_mutex);
            return res;
        }
        xSemaphoreGive(snapshot_mutex);
    }
    
    // Snapshot yoksa hata döndür
    httpd_resp_send_404(req);
    return ESP_FAIL;
}

static esp_err_t flash_handler(httpd_req_t *req) {
    char buf[10], state[2];
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        if (httpd_query_key_value(buf, "state", state, sizeof(state)) == ESP_OK) {
            setFlash(state[0] == '1');
        }
    }
    return httpd_resp_send(req, "OK", 2);
}

// Threshold ayarları handler
static esp_err_t threshold_handler(httpd_req_t *req) {
    char buf[64];
    char param[8];
    
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        // Threshold değeri
        if (httpd_query_key_value(buf, "val", param, sizeof(param)) == ESP_OK) {
            g_threshold = constrain(atoi(param), 0, 255);
        }
        // Aktif/Pasif
        if (httpd_query_key_value(buf, "en", param, sizeof(param)) == ESP_OK) {
            g_threshold_enabled = (param[0] == '1');
        }
        // Invert
        if (httpd_query_key_value(buf, "inv", param, sizeof(param)) == ESP_OK) {
            g_invert = (param[0] == '1');
        }
    }
    
    char json[128];
    snprintf(json, sizeof(json), "{\"threshold\":%d,\"enabled\":%s,\"invert\":%s}",
             g_threshold, g_threshold_enabled ? "true" : "false", g_invert ? "true" : "false");
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// DEBUG: Preprocessed 96x96 görüntüyü göster (threshold uygulanmış)
static uint8_t debug_preprocessed[96*96];  // Son preprocess edilen görüntü

static esp_err_t debug_input_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // Preprocess yap ve debug buffer'a kaydet (GRAYSCALE format)
    uint8_t* img = fb->buf;  // Grayscale: 1 byte per pixel
    int src_w = fb->width;
    int src_h = fb->height;
    float x_ratio = (float)src_w / 96;
    float y_ratio = (float)src_h / 96;
    
    uint8_t min_val = 255, max_val = 0;
    
    // First pass: resize and find min/max
    for (int y = 0; y < 96; y++) {
        for (int x = 0; x < 96; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            uint8_t val = img[src_y * src_w + src_x];
            debug_preprocessed[y * 96 + x] = val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    esp_camera_fb_return(fb);
    
    // Second pass: contrast stretch + threshold
    int range = max_val - min_val;
    if (range < 10) range = 10;
    
    for (int i = 0; i < 96*96; i++) {
        int stretched = ((debug_preprocessed[i] - min_val) * 255) / range;
        stretched = constrain(stretched, 0, 255);
        
        // Apply threshold if enabled
        if (g_threshold_enabled) {
            if (g_invert) {
                stretched = (stretched < g_threshold) ? 255 : 0;
            } else {
                stretched = (stretched >= g_threshold) ? 255 : 0;
            }
        }
        debug_preprocessed[i] = stretched;
    }
    
    // Create simple BMP header for 96x96 grayscale
    static uint8_t bmp_header[1078];  // 54 + 1024 (palette)
    memset(bmp_header, 0, sizeof(bmp_header));
    
    // BMP Header
    bmp_header[0] = 'B'; bmp_header[1] = 'M';
    uint32_t filesize = 1078 + 96*96;
    memcpy(&bmp_header[2], &filesize, 4);
    uint32_t offset = 1078;
    memcpy(&bmp_header[10], &offset, 4);
    
    // DIB Header
    uint32_t dibsize = 40;
    memcpy(&bmp_header[14], &dibsize, 4);
    int32_t width = 96, height = -96;  // Negative for top-down
    memcpy(&bmp_header[18], &width, 4);
    memcpy(&bmp_header[22], &height, 4);
    uint16_t planes = 1;
    memcpy(&bmp_header[26], &planes, 2);
    uint16_t bpp = 8;
    memcpy(&bmp_header[28], &bpp, 2);
    uint32_t imgsize = 96*96;
    memcpy(&bmp_header[34], &imgsize, 4);
    
    // Grayscale palette
    for (int i = 0; i < 256; i++) {
        bmp_header[54 + i*4 + 0] = i;  // B
        bmp_header[54 + i*4 + 1] = i;  // G
        bmp_header[54 + i*4 + 2] = i;  // R
        bmp_header[54 + i*4 + 3] = 0;  // Reserved
    }
    
    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send_chunk(req, (const char*)bmp_header, 1078);
    httpd_resp_send_chunk(req, (const char*)debug_preprocessed, 96*96);
    return httpd_resp_send_chunk(req, NULL, 0);
}

void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 10;
    config.stack_size = 16384;  // Stack overflow için artırıldı (default: 4096)
    
    httpd_uri_t index_uri = { "/", HTTP_GET, index_handler, NULL };
    httpd_uri_t detect_uri = { "/detect", HTTP_GET, detect_handler, NULL };
    httpd_uri_t capture_uri = { "/capture", HTTP_GET, capture_handler, NULL };
    httpd_uri_t snapshot_uri = { "/snapshot", HTTP_GET, snapshot_handler, NULL };
    httpd_uri_t flash_uri = { "/flash", HTTP_GET, flash_handler, NULL };
    httpd_uri_t debug_uri = { "/debug_input", HTTP_GET, debug_input_handler, NULL };
    httpd_uri_t threshold_uri = { "/threshold", HTTP_GET, threshold_handler, NULL };
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &detect_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        httpd_register_uri_handler(camera_httpd, &snapshot_uri);
        httpd_register_uri_handler(camera_httpd, &flash_uri);
        httpd_register_uri_handler(camera_httpd, &debug_uri);
        httpd_register_uri_handler(camera_httpd, &threshold_uri);
    }
    
    config.server_port = 81;
    config.ctrl_port = 32769;
    httpd_uri_t stream_uri = { "/stream", HTTP_GET, stream_handler, NULL };
    
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    }
    
    Serial.println("Web sunucu hazir!");
}

// ==================== SETUP & LOOP ====================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    Serial.begin(115200);
    Serial.println("\n==============================");
    Serial.println("EE4065 - Digit Detection");
    Serial.println("==============================\n");
    
    // Snapshot mutex oluştur
    snapshot_mutex = xSemaphoreCreateMutex();
    
    pinMode(LED_GPIO_NUM, OUTPUT);
    setFlash(false);
    
    if (!initCamera()) {
        Serial.println("HATA: Kamera başlatılamadı!");
        while(1) delay(1000);
    }
    
    if (!initTFLite()) {
        Serial.println("HATA: TFLite başlatılamadı!");
        while(1) delay(1000);
    }
    
    // WiFi başlat (AP veya Station modunda)
    #if USE_AP_MODE
        // AP Mode - ESP32 kendi WiFi'sini oluşturur
        WiFi.mode(WIFI_AP);
        WiFi.softAP(ap_ssid, ap_password);
        delay(100);
        
        IPAddress IP = WiFi.softAPIP();
        Serial.println("\n========== AP MODE ==========");
        Serial.printf("SSID: %s\n", ap_ssid);
        Serial.printf("Sifre: %s\n", ap_password);
        Serial.printf("IP: %s\n", IP.toString().c_str());
        Serial.println("=============================");
        startServer();
    #else
        // Station Mode - mevcut WiFi'ye bağlan
        WiFi.mode(WIFI_STA);
        WiFi.disconnect(true);  // Önceki bağlantıları temizle
        delay(100);
        
        Serial.printf("\nWiFi SSID: %s\n", sta_ssid);
        Serial.print("Bağlanıyor");
        
        WiFi.begin(sta_ssid, sta_password);
        
        int retry = 0;
        int max_retry = 60;  // 30 saniye timeout (60 * 500ms)
        
        while (WiFi.status() != WL_CONNECTED && retry < max_retry) {
            delay(500);
            Serial.print(".");
            retry++;
            
            // Her 10 denemede durum göster
            if (retry % 10 == 0) {
                Serial.printf(" (%d/%d)", retry, max_retry);
            }
        }
        
        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("\n========== BAGLANDI ==========");
            Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
            Serial.printf("RSSI: %d dBm\n", WiFi.RSSI());
            Serial.println("===============================");
            startServer();
        } else {
            // Bağlanamadı - AP moduna geç
            Serial.println("\n\nStation mode basarisiz! AP moduna geciliyor...");
            WiFi.disconnect(true);
            delay(100);
            
            WiFi.mode(WIFI_AP);
            WiFi.softAP(ap_ssid, ap_password);
            delay(100);
            
            IPAddress IP = WiFi.softAPIP();
            Serial.println("\n========== AP MODE (Fallback) ==========");
            Serial.printf("SSID: %s\n", ap_ssid);
            Serial.printf("Sifre: %s\n", ap_password);
            Serial.printf("IP: %s\n", IP.toString().c_str());
            Serial.println("=========================================");
            startServer();
        }
    #endif
    
    Serial.println("\nSistem hazır! Komutlar: d=tespit, f=flash, i=info\n");
}

void loop() {
    // Station modunda bağlantı kontrolü
    #if !USE_AP_MODE
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.reconnect();
            delay(5000);
        }
    #endif
    
    if (Serial.available()) {
        char cmd = Serial.read();
        
        if (cmd == 'd' || cmd == 'D') {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) {
                detectDigits(fb);
                esp_camera_fb_return(fb);
            }
        }
        else if (cmd == 'f' || cmd == 'F') {
            static bool f = false;
            f = !f;
            setFlash(f);
            Serial.printf("Flash: %s\n", f ? "AÇIK" : "KAPALI");
        }
        else if (cmd == 'i' || cmd == 'I') {
            #if USE_AP_MODE
                Serial.printf("Heap: %d, PSRAM: %d, AP IP: %s, Clients: %d\n",
                              ESP.getFreeHeap(), ESP.getFreePsram(),
                              WiFi.softAPIP().toString().c_str(),
                              WiFi.softAPgetStationNum());
            #else
                Serial.printf("Heap: %d, PSRAM: %d, WiFi: %s\n",
                              ESP.getFreeHeap(), ESP.getFreePsram(),
                              WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "Yok");
            #endif
        }
    }
    
    delay(10);
}
