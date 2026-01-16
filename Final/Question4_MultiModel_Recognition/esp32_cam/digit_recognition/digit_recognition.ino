/*
 * Question 4: Multi-Model Handwritten Digit Recognition
 * EE4065 Final Project - ESP32-CAM
 * 
 * Supports: SqueezeNetMini, MobileNetV2Mini, ResNet8, EfficientNetMini
 * Web UI with model selection buttons
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <esp_http_server.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// TensorFlow Lite Micro
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Headers
#include "squeezenetmini_model.h"
#include "mobilenetv2mini_model.h"
#include "resnet8_model.h"
#include "efficientnetmini_model.h"

// ==================== CONFIGURATION ====================
#define USE_AP_MODE true

const char* ap_ssid = "ESP32-Digit-Q4";
const char* ap_password = "12345678";

const char* sta_ssid = "DESKTOP-GD3K1J9 0614";
const char* sta_password = "108108108";

// Camera pins (AI-THINKER)
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

// Model IDs
#define MODEL_SQUEEZENET    0
#define MODEL_MOBILENET     1
#define MODEL_RESNET        2
#define MODEL_EFFICIENTNET  3
#define MODEL_FUSION        4

// Input/Output sizes
#define INPUT_SIZE 32
#define INPUT_CHANNELS 3
#define NUM_CLASSES 10

// ==================== GLOBALS ====================
httpd_handle_t camera_httpd = NULL;
const char* digit_labels[10] = {"0","1","2","3","4","5","6","7","8","9"};

// TFLite variables for each model
constexpr int TENSOR_ARENA_SIZE = 90 * 1024;  // 90KB for largest model
// Tensor arena will be allocated in PSRAM (setup)
uint8_t* tensor_arena = nullptr;

const tflite::Model* current_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Current model info
int current_model_id = MODEL_SQUEEZENET;
const char* model_names[] = {"SqueezeNet", "MobileNet", "ResNet8", "EfficientNet", "Fusion"};

// Last prediction results
float last_probs[5][NUM_CLASSES];  // Store probs for each model
int last_predictions[5];
float last_confidences[5];
unsigned long last_inference_time[5];

// Debug buffer - stores preprocessed grayscale for visualization
uint8_t debug_image[INPUT_SIZE * INPUT_SIZE];

// Preprocessing buffers in PSRAM (allocated in setup)
#define WORK_SIZE 120  // Matches 160x120 camera height
uint8_t* preprocess_gray = nullptr;   // WORK_SIZE * WORK_SIZE
uint8_t* preprocess_bin = nullptr;    // WORK_SIZE * WORK_SIZE  
uint8_t* preprocess_dilated = nullptr; // WORK_SIZE * WORK_SIZE

// ==================== THRESHOLD SETTINGS (from Q2) ====================
int g_threshold = 128;        // Binary threshold value (0-255)
bool g_use_threshold = true;  // Enable/disable thresholding

// ==================== MODEL LOADING ====================
const unsigned char* getModelData(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_model;
        case MODEL_MOBILENET:    return mobilenetv2mini_model;
        case MODEL_RESNET:       return resnet8_model;
        case MODEL_EFFICIENTNET: return efficientnetmini_model;
        default: return squeezenetmini_model;
    }
}

unsigned int getModelLen(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_model_len;
        case MODEL_MOBILENET:    return mobilenetv2mini_model_len;
        case MODEL_RESNET:       return resnet8_model_len;
        case MODEL_EFFICIENTNET: return efficientnetmini_model_len;
        default: return squeezenetmini_model_len;
    }
}

float getInputScale(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_input_scale;
        case MODEL_MOBILENET:    return mobilenetv2mini_input_scale;
        case MODEL_RESNET:       return resnet8_input_scale;
        case MODEL_EFFICIENTNET: return efficientnetmini_input_scale;
        default: return 0.003921569f;
    }
}

int getInputZeroPoint(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_input_zero_point;
        case MODEL_MOBILENET:    return mobilenetv2mini_input_zero_point;
        case MODEL_RESNET:       return resnet8_input_zero_point;
        case MODEL_EFFICIENTNET: return efficientnetmini_input_zero_point;
        default: return 0;
    }
}

float getOutputScale(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_output_scale;
        case MODEL_MOBILENET:    return mobilenetv2mini_output_scale;
        case MODEL_RESNET:       return resnet8_output_scale;
        case MODEL_EFFICIENTNET: return efficientnetmini_output_scale;
        default: return 0.00390625f;
    }
}

int getOutputZeroPoint(int model_id) {
    switch(model_id) {
        case MODEL_SQUEEZENET:   return squeezenetmini_output_zero_point;
        case MODEL_MOBILENET:    return mobilenetv2mini_output_zero_point;
        case MODEL_RESNET:       return resnet8_output_zero_point;
        case MODEL_EFFICIENTNET: return efficientnetmini_output_zero_point;
        default: return -128;
    }
}

// Op Resolver with operators needed for CNN models
static tflite::MicroMutableOpResolver<20> resolver;

bool initTFLite(int model_id) {
    Serial.printf("Loading model: %s\n", model_names[model_id]);
    
    const unsigned char* model_data = getModelData(model_id);
    current_model = tflite::GetModel(model_data);
    
    if (current_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }
    
    // Register ops (only once)
    static bool ops_registered = false;
    if (!ops_registered) {
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddMaxPool2D();
        resolver.AddAveragePool2D();
        resolver.AddReshape();
        resolver.AddSoftmax();
        resolver.AddRelu();
        resolver.AddRelu6();
        resolver.AddAdd();
        resolver.AddMul();
        resolver.AddMean();
        resolver.AddPad();
        resolver.AddConcatenation();
        resolver.AddQuantize();
        resolver.AddDequantize();
        resolver.AddLogistic();
        resolver.AddFullyConnected(); // <--- ADDED THIS
        ops_registered = true;
    }
    
    // Create interpreter
    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::MicroInterpreter static_interp(
        current_model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interpreter = &static_interp;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return false;
    }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    Serial.printf("  Input: [%d,%d,%d,%d]\n", 
        input_tensor->dims->data[0], input_tensor->dims->data[1],
        input_tensor->dims->data[2], input_tensor->dims->data[3]);
    Serial.printf("  Output: [%d,%d]\n",
        output_tensor->dims->data[0], output_tensor->dims->data[1]);
    Serial.printf("  Arena used: %d bytes\n", interpreter->arena_used_bytes());
    
    current_model_id = model_id;
    return true;
}

// ==================== CAMERA ====================
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
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    
    Serial.println("Camera initialized");
    return true;
}

// ==================== CONNECTED COMPONENT (from Q1b) ====================
typedef struct { int x, y; } CCPoint;
CCPoint ccStack[2000];  // Stack for flood fill

int floodFillCC(uint8_t *binary, uint8_t *visited, int w, int h, int startX, int startY) {
    int area = 0;
    int stackPtr = 0;
    
    ccStack[stackPtr++] = (CCPoint){startX, startY};
    
    while (stackPtr > 0) {
        CCPoint p = ccStack[--stackPtr];
        int x = p.x, y = p.y;
        
        if (x < 0 || x >= w || y < 0 || y >= h) continue;
        
        int idx = y * w + x;
        if (visited[idx] || binary[idx] == 0) continue;
        
        visited[idx] = 1;
        area++;
        
        // 4-connectivity
        if (stackPtr < 1996) {
            ccStack[stackPtr++] = (CCPoint){x+1, y};
            ccStack[stackPtr++] = (CCPoint){x-1, y};
            ccStack[stackPtr++] = (CCPoint){x, y+1};
            ccStack[stackPtr++] = (CCPoint){x, y-1};
        }
    }
    
    return area;
}

// Find and keep only the largest connected component
void keepLargestComponent(uint8_t *binary, int w, int h) {
    uint8_t *labels = (uint8_t*)calloc(w * h, 1);
    if (!labels) return;
    
    int largestArea = 0;
    int largestStartX = 0, largestStartY = 0;
    
    // First pass: find largest
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (binary[idx] == 255 && !labels[idx]) {
                int area = floodFillCC(binary, labels, w, h, x, y);
                if (area > largestArea) {
                    largestArea = area;
                    largestStartX = x;
                    largestStartY = y;
                }
            }
        }
    }
    
    // Second pass: keep only largest
    memset(labels, 0, w * h);
    
    // Zero out binary first
    uint8_t *newBinary = (uint8_t*)calloc(w * h, 1);
    if (newBinary) {
        // Flood fill again to mark largest component
        int stackPtr = 0;
        ccStack[stackPtr++] = (CCPoint){largestStartX, largestStartY};
        
        while (stackPtr > 0) {
            CCPoint p = ccStack[--stackPtr];
            int x = p.x, y = p.y;
            
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            
            int idx = y * w + x;
            if (labels[idx] || binary[idx] == 0) continue;
            
            labels[idx] = 1;
            newBinary[idx] = 255;
            
            if (stackPtr < 1996) {
                ccStack[stackPtr++] = (CCPoint){x+1, y};
                ccStack[stackPtr++] = (CCPoint){x-1, y};
                ccStack[stackPtr++] = (CCPoint){x, y+1};
                ccStack[stackPtr++] = (CCPoint){x, y-1};
            }
        }
        
        memcpy(binary, newBinary, w * h);
        free(newBinary);
    }
    
    free(labels);
    Serial.printf("Largest component: %d pixels\n", largestArea);
}

// ==================== PREPROCESSING (GRAYSCALE - SIMPLE!) ====================
void preprocessImage(camera_fb_t* fb, int8_t* data, int model_id) {
    uint8_t* img = fb->buf;  // GRAYSCALE = direct uint8_t bytes!
    int src_w = fb->width;   // 160
    int src_h = fb->height;  // 120

    float input_scale = getInputScale(model_id);
    int input_zp = getInputZeroPoint(model_id);

    // --- STEP 1: Center Crop 120x120 from 160x120 ---
    int off_x = (src_w - WORK_SIZE) / 2; // (160-120)/2 = 20
    int off_y = 0;

    if (src_w < WORK_SIZE || src_h < WORK_SIZE) {
        Serial.println("ERR: Frame too small!");
        return; 
    }
    
    uint8_t min_val = 255, max_val = 0;
    
    for (int y = 0; y < WORK_SIZE; y++) {
        for (int x = 0; x < WORK_SIZE; x++) {
            int sx = off_x + x;
            int sy = off_y + y;
            
            // GRAYSCALE: Direct read - no conversion needed!
            uint8_t gray = img[sy * src_w + sx];
            
            preprocess_gray[y * WORK_SIZE + x] = gray;
            if (gray < min_val) min_val = gray;
            if (gray > max_val) max_val = gray;
        }
    }

    // --- STEP 2: CONTRAST STRETCH ---
    int range = max_val - min_val;
    if (range < 10) range = 10;
    
    for (int i = 0; i < WORK_SIZE * WORK_SIZE; i++) {
        int stretched = ((preprocess_gray[i] - min_val) * 255) / range;
        preprocess_gray[i] = (stretched < 0) ? 0 : (stretched > 255) ? 255 : stretched;
    }

    // --- STEP 3: BINARY THRESHOLD ---
    // Dark digit on light background: dark pixels (< threshold) = 255 (foreground)
    int threshold = g_threshold;
    
    if (g_use_threshold) {
        for (int i = 0; i < WORK_SIZE * WORK_SIZE; i++) {
            preprocess_bin[i] = (preprocess_gray[i] > threshold) ? 255 : 0;
        }
    } else {
        memcpy(preprocess_bin, preprocess_gray, WORK_SIZE * WORK_SIZE);
    }

    // --- STEP 4: CONNECTED COMPONENT - KEEP ONLY LARGEST (THE DIGIT!) ---
    if (g_use_threshold) {
        keepLargestComponent(preprocess_bin, WORK_SIZE, WORK_SIZE);
    }

    // --- STEP 5: DILATION (make digit thicker) ---
    memcpy(preprocess_dilated, preprocess_bin, WORK_SIZE * WORK_SIZE);
    
    for (int y = 1; y < WORK_SIZE - 1; y++) {
        for (int x = 1; x < WORK_SIZE - 1; x++) {
            int i = y * WORK_SIZE + x;
            if (preprocess_bin[i] || preprocess_bin[i-1] || preprocess_bin[i+1] ||
                preprocess_bin[i-WORK_SIZE] || preprocess_bin[i+WORK_SIZE]) {
                preprocess_dilated[i] = 255;
            }
        }
    }
    memcpy(preprocess_bin, preprocess_dilated, WORK_SIZE * WORK_SIZE);

    // --- STEP 6: Find Bounding Box of the digit ---
    int minx = WORK_SIZE, miny = WORK_SIZE, maxx = -1, maxy = -1;
    int pixels = 0;

    for (int y = 0; y < WORK_SIZE; y++) {
        for (int x = 0; x < WORK_SIZE; x++) {
            if (preprocess_bin[y * WORK_SIZE + x]) {
                if (x < minx) minx = x;
                if (x > maxx) maxx = x;
                if (y < miny) miny = y;
                if (y > maxy) maxy = y;
                pixels++;
            }
        }
    }

    if (pixels < 10) {
        // No digit found, use full frame
        minx = 0; miny = 0; maxx = WORK_SIZE - 1; maxy = WORK_SIZE - 1;
    }

    // Add Padding (15%)
    int pad_x = (maxx - minx + 1) * 15 / 100;
    int pad_y = (maxy - miny + 1) * 15 / 100;
    minx = max(0, minx - pad_x);
    miny = max(0, miny - pad_y);
    maxx = min(WORK_SIZE - 1, maxx + pad_x);
    maxy = min(WORK_SIZE - 1, maxy + pad_y);

    int box_w = maxx - minx + 1;
    int box_h = maxy - miny + 1;

    // --- STEP 7: Resize to 32x32 for model input ---
    for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
            int sx = minx + (x * box_w) / INPUT_SIZE;
            int sy = miny + (y * box_h) / INPUT_SIZE;
            
            if (sx < 0) sx = 0; if (sx >= WORK_SIZE) sx = WORK_SIZE - 1;
            if (sy < 0) sy = 0; if (sy >= WORK_SIZE) sy = WORK_SIZE - 1;

            uint8_t final_val = preprocess_bin[sy * WORK_SIZE + sx];

            debug_image[y * INPUT_SIZE + x] = final_val;

            float v = final_val / 255.0f;
            int8_t q = (int8_t)(v / input_scale + input_zp);

            data[(y * INPUT_SIZE + x) * 3 + 0] = q;
            data[(y * INPUT_SIZE + x) * 3 + 1] = q;
            data[(y * INPUT_SIZE + x) * 3 + 2] = q;
        }
    }
    
    Serial.printf("Digit box: [%d,%d]-[%d,%d] (%d pixels)\n", minx, miny, maxx, maxy, pixels);
}

// ==================== INFERENCE ====================
// GitHub finding: Flush old frames before capture (jomjol/AI-on-the-edge-device)
int runInference(int model_id) {
    // Flush any old buffered frames (prevents stuck predictions)
    camera_fb_t* flush_fb = esp_camera_fb_get();
    if (flush_fb) {
        esp_camera_fb_return(flush_fb);
    }
    
    // Get fresh frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return -1;
    }
    
    // Reload model if needed
    if (current_model_id != model_id) {
        if (!initTFLite(model_id)) {
            esp_camera_fb_return(fb);
            return -1;
        }
    }
    
    // Preprocess image
    preprocessImage(fb, input_tensor->data.int8, model_id);
    esp_camera_fb_return(fb);
    
    // Run inference
    unsigned long start = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed!");
        return -1;
    }
    last_inference_time[model_id] = millis() - start;
    
    // Get output
    float output_scale = getOutputScale(model_id);
    int output_zp = getOutputZeroPoint(model_id);
    
    // Dequantize outputs
    float logits[NUM_CLASSES];
    float max_logit = -1000;
    
    Serial.print("Raw output: ");
    for (int i = 0; i < NUM_CLASSES; i++) {
        logits[i] = (output_tensor->data.int8[i] - output_zp) * output_scale;
        if (logits[i] > max_logit) max_logit = logits[i];
        Serial.printf("%.2f ", logits[i]);
    }
    Serial.println();
    
    // Apply softmax for proper probabilities
    float sum_exp = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        logits[i] = expf(logits[i] - max_logit);  // Subtract max for numerical stability
        sum_exp += logits[i];
    }
    
    int best_class = 0;
    float best_prob = 0;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        float prob = logits[i] / sum_exp;  // Softmax
        last_probs[model_id][i] = prob;
        if (prob > best_prob) {
            best_prob = prob;
            best_class = i;
        }
    }
    
    last_predictions[model_id] = best_class;
    last_confidences[model_id] = best_prob;
    
    Serial.printf("%s: predicted %d (%.1f%%) in %lu ms\n",
        model_names[model_id], best_class, best_prob * 100, last_inference_time[model_id]);
    
    return best_class;
}

// Run all models and fuse results
int runFusion() {
    Serial.println("Running fusion...");
    
    // Run each model
    for (int i = 0; i < 4; i++) {
        runInference(i);
    }
    
    // Average probabilities
    float fused_probs[NUM_CLASSES] = {0};
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int m = 0; m < 4; m++) {
            fused_probs[c] += last_probs[m][c];
        }
        fused_probs[c] /= 4.0f;
    }
    
    // Find best class
    int best_class = 0;
    float best_prob = fused_probs[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (fused_probs[i] > best_prob) {
            best_prob = fused_probs[i];
            best_class = i;
        }
    }
    
    // Store fusion results
    for (int i = 0; i < NUM_CLASSES; i++) {
        last_probs[MODEL_FUSION][i] = fused_probs[i];
    }
    last_predictions[MODEL_FUSION] = best_class;
    last_confidences[MODEL_FUSION] = best_prob;
    last_inference_time[MODEL_FUSION] = 
        last_inference_time[0] + last_inference_time[1] + 
        last_inference_time[2] + last_inference_time[3];
    
    Serial.printf("FUSION: predicted %d (%.1f%%) total time %lu ms\n",
        best_class, best_prob * 100, last_inference_time[MODEL_FUSION]);
    
    return best_class;
}

// ==================== WEB HANDLERS ====================
const char* HTML_PAGE = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Q4: Multi-Model Digit Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee; 
            min-height: 100vh; 
            padding: 20px;
        }
        h1 { 
            text-align: center; 
            color: #00d9ff; 
            margin-bottom: 10px;
            text-shadow: 0 0 10px rgba(0,217,255,0.5);
        }
        h2 { 
            text-align: center; 
            color: #888; 
            font-weight: normal;
            margin-bottom: 30px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h3 {
            color: #00d9ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .model-buttons {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        .model-btn {
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .model-btn:hover { transform: scale(1.05); }
        .btn-squeeze { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .btn-mobile { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }
        .btn-resnet { background: linear-gradient(135deg, #ee0979, #ff6a00); color: white; }
        .btn-efficient { background: linear-gradient(135deg, #fc4a1a, #f7b733); color: white; }
        .btn-fusion {
            grid-column: span 2;
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            font-size: 16px;
        }
        #result {
            font-size: 72px;
            text-align: center;
            padding: 30px;
            background: rgba(0,217,255,0.1);
            border-radius: 15px;
            margin: 20px 0;
        }
        #confidence {
            text-align: center;
            font-size: 24px;
            color: #00ff88;
        }
        #model-name {
            text-align: center;
            color: #888;
            margin-top: 10px;
        }
        .probs-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 5px;
            margin-top: 20px;
        }
        .prob-item {
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .prob-label { font-size: 24px; }
        .prob-value { font-size: 12px; color: #888; }
        #status {
            text-align: center;
            color: #888;
            margin-top: 20px;
        }
        .all-results {
            margin-top: 20px;
        }
        .model-result {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 5px;
        }
        .loading { animation: pulse 1s infinite; }
        @keyframes pulse { 
            0%, 100% { opacity: 1; } 
            50% { opacity: 0.5; } 
        }
    </style>
</head>
<body>
    <h1>🔢 Multi-Model Digit Recognition</h1>
    <h2>EE4065 Final Project - Question 4</h2>
    
    <div class="container">
        <div class="panel">
            <h3>📷 Camera Preview</h3>
            <div style="text-align:center; margin-bottom:15px;">
                <div style="display:flex; justify-content:center; gap:10px; flex-wrap:wrap;">
                    <div>
                        <div style="color:#888; font-size:12px; margin-bottom:5px;">Camera</div>
                        <img id="camera-img" src="" style="width:160px; height:160px; border-radius:10px; border:2px solid #333; object-fit:cover;" onclick="refreshImage()">
                    </div>
                    <div>
                        <div style="color:#888; font-size:12px; margin-bottom:5px;">Model Input (32×32)</div>
                        <img id="debug-img" src="" style="width:160px; height:160px; border-radius:10px; border:2px solid #00d9ff;">
                    </div>
                </div>
                <div style="margin-top:10px;">
                    <button class="model-btn" style="background:#444; padding:10px 20px;" onclick="refreshImage()">🔄 Refresh</button>
                    <button class="model-btn" style="background:#f90; padding:10px 20px;" id="flash-btn" onclick="toggleFlash()">💡 Flash: OFF</button>
                    <button class="model-btn" style="background:linear-gradient(135deg,#ff4444,#cc0000); padding:10px 20px;" onclick="resetESP32()">♻️ Reset</button>
                </div>
            </div>
            
            <h3>🤖 Select Model</h3>
            <div class="model-buttons">
                <button class="model-btn btn-squeeze" onclick="runModel(0)">SqueezeNet</button>
                <button class="model-btn btn-mobile" onclick="runModel(1)">MobileNet</button>
                <button class="model-btn btn-resnet" onclick="runModel(2)">ResNet8</button>
                <button class="model-btn btn-efficient" onclick="runModel(3)">EfficientNet</button>
                <button class="model-btn btn-fusion" onclick="runModel(4)">🔗 Run All & Fuse</button>
            </div>
            
            <div id="result">-</div>
            <div id="confidence">Hold a digit in front of camera</div>
            <div id="model-name"></div>
            
            <div class="probs-grid" id="probs">
                <div class="prob-item"><div class="prob-label">0</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">1</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">2</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">3</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">4</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">5</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">6</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">7</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">8</div><div class="prob-value">-</div></div>
                <div class="prob-item"><div class="prob-label">9</div><div class="prob-value">-</div></div>
            </div>
            
            <!-- THRESHOLD CONTROLS (ported from Q2) -->
            <div style="margin-top:20px; padding:15px; background:rgba(255,255,255,0.05); border-radius:10px;">
                <h3 style="color:#ff9800; margin-bottom:10px;">🎚️ Image Processing</h3>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <input type="checkbox" id="thresh-enable" checked onchange="updateThreshold()">
                    <label for="thresh-enable" style="cursor:pointer;">Enable Threshold</label>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span>Threshold:</span>
                    <input type="range" id="thresh-slider" min="0" max="255" value="128" style="flex:1;" onchange="updateThreshold()">
                    <span id="thresh-value">128</span>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h3>📊 All Model Results</h3>
            <div class="all-results" id="all-results">
                <div class="model-result">
                    <span>SqueezeNet</span>
                    <span id="res-0">-</span>
                </div>
                <div class="model-result">
                    <span>MobileNet</span>
                    <span id="res-1">-</span>
                </div>
                <div class="model-result">
                    <span>ResNet8</span>
                    <span id="res-2">-</span>
                </div>
                <div class="model-result">
                    <span>EfficientNet</span>
                    <span id="res-3">-</span>
                </div>
                <div class="model-result" style="background: rgba(0,217,255,0.2);">
                    <span><strong>FUSION</strong></span>
                    <span id="res-4">-</span>
                </div>
            </div>
            <div id="status"></div>
        </div>
    </div>
    
    <script>
        const modelNames = ['SqueezeNet', 'MobileNet', 'ResNet8', 'EfficientNet', 'Fusion'];
        
        function refreshImage() {
            document.getElementById('camera-img').src = '/snapshot?' + Date.now();
        }
        
        // Auto-load image on page load
        window.onload = function() {
            refreshImage();
        };
        
        let flashOn = false;
        async function toggleFlash() {
            flashOn = !flashOn;
            await fetch('/flash?on=' + (flashOn ? '1' : '0'));
            document.getElementById('flash-btn').textContent = '💡 Flash: ' + (flashOn ? 'ON' : 'OFF');
            document.getElementById('flash-btn').style.background = flashOn ? '#0f0' : '#f90';
            refreshImage();
        }
        
        async function resetESP32() {
            document.getElementById('status').textContent = 'Resetting...';
            document.getElementById('result').textContent = '-';
            document.getElementById('confidence').textContent = 'Resetting ESP32...';
            try {
                const resp = await fetch('/reset');
                if (resp.ok) {
                    document.getElementById('status').textContent = 'Reset complete';
                    document.getElementById('confidence').textContent = 'Hold a digit in front of camera';
                    refreshImage();
                    document.getElementById('debug-img').src = '/debug_input?' + Date.now();
                }
            } catch (e) {
                document.getElementById('status').textContent = 'Reset failed';
            }
        }
        
        async function runModel(id) {
            document.getElementById('result').textContent = '...';
            document.getElementById('result').classList.add('loading');
            document.getElementById('confidence').textContent = 'Running ' + modelNames[id] + '...';
            document.getElementById('status').textContent = 'Processing...';
            
            try {
                const resp = await fetch('/predict?model=' + id);
                const data = await resp.json();
                
                document.getElementById('result').textContent = data.prediction;
                document.getElementById('result').classList.remove('loading');
                document.getElementById('confidence').textContent = 
                    (data.confidence * 100).toFixed(1) + '% confidence';
                document.getElementById('model-name').textContent = 
                    modelNames[id] + ' - ' + data.time + 'ms';
                
                // Update probability bars
                const probsDiv = document.getElementById('probs');
                const items = probsDiv.querySelectorAll('.prob-item');
                for (let i = 0; i < 10; i++) {
                    const val = (data.probs[i] * 100).toFixed(1);
                    items[i].querySelector('.prob-value').textContent = val + '%';
                    items[i].style.background = i == data.prediction ? 
                        'rgba(0,255,136,0.3)' : 'rgba(255,255,255,0.05)';
                }
                
                // Update model result
                document.getElementById('res-' + id).textContent = 
                    data.prediction + ' (' + (data.confidence * 100).toFixed(0) + '%)';
                
                document.getElementById('status').textContent = 
                    'Completed in ' + data.time + 'ms';
                
                // Refresh camera and debug images after prediction
                refreshImage();
                document.getElementById('debug-img').src = '/debug_input?' + Date.now();
                    
            } catch (e) {
                document.getElementById('result').textContent = 'Error';
                document.getElementById('result').classList.remove('loading');
                document.getElementById('confidence').textContent = e.toString();
            }
        }
        
        // Threshold control function (ported from Q2)
        async function updateThreshold() {
            const enabled = document.getElementById('thresh-enable').checked;
            const value = document.getElementById('thresh-slider').value;
            document.getElementById('thresh-value').textContent = value;
            
            try {
                await fetch('/threshold?value=' + value + '&enable=' + (enabled ? '1' : '0'));
                document.getElementById('status').textContent = 'Threshold: ' + (enabled ? value : 'OFF');
                // Refresh debug image to show effect
                document.getElementById('debug-img').src = '/debug_input?' + Date.now();
            } catch (e) {
                console.error('Threshold update failed:', e);
            }
        }
    </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, HTML_PAGE, strlen(HTML_PAGE));
}

static esp_err_t predict_handler(httpd_req_t *req) {
    char query[32];
    int model_id = 0;
    
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[8];
        if (httpd_query_key_value(query, "model", val, sizeof(val)) == ESP_OK) {
            model_id = atoi(val);
        }
    }
    
    int prediction;
    if (model_id == MODEL_FUSION) {
        prediction = runFusion();
    } else {
        prediction = runInference(model_id);
    }
    
    // Build JSON response
    char json[512];
    int off = snprintf(json, sizeof(json),
        "{\"prediction\":%d,\"confidence\":%.4f,\"time\":%lu,\"probs\":[",
        last_predictions[model_id],
        last_confidences[model_id],
        last_inference_time[model_id]);
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        off += snprintf(json + off, sizeof(json) - off, "%.4f%s",
            last_probs[model_id][i], i < 9 ? "," : ""
        );
    }
    snprintf(json + off, sizeof(json) - off, "]}");
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// Snapshot handler - returns camera image as JPEG
// GitHub finding: Flush old frame before capture for fresh image
static esp_err_t snapshot_handler(httpd_req_t *req) {
    // Flush any buffered frame first
    camera_fb_t *flush_fb = esp_camera_fb_get();
    if (flush_fb) esp_camera_fb_return(flush_fb);
    
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // Convert RGB565 to JPEG
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
    esp_camera_fb_return(fb);
    
    if (!converted) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char*)jpg_buf, jpg_len);
    free(jpg_buf);
    
    return ESP_OK;
}

// Debug handler - returns preprocessed 32x32 image as scaled BMP
static esp_err_t debug_handler(httpd_req_t *req) {
    // Scale up to 160x160 for visibility (5x)
    const int scale = 5;
    const int out_size = INPUT_SIZE * scale;  // 160
    const int row_bytes = ((out_size * 3 + 3) / 4) * 4;  // Padded to 4 bytes
    const int data_size = row_bytes * out_size;
    const int file_size = 54 + data_size;
    
    uint8_t* bmp = (uint8_t*)malloc(file_size);
    if (!bmp) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // BMP Header
    memset(bmp, 0, 54);
    bmp[0] = 'B'; bmp[1] = 'M';
    bmp[2] = file_size & 0xFF;
    bmp[3] = (file_size >> 8) & 0xFF;
    bmp[4] = (file_size >> 16) & 0xFF;
    bmp[5] = (file_size >> 24) & 0xFF;
    bmp[10] = 54;  // Data offset
    bmp[14] = 40;  // DIB header size
    bmp[18] = out_size & 0xFF;
    bmp[19] = (out_size >> 8) & 0xFF;
    bmp[22] = out_size & 0xFF;
    bmp[23] = (out_size >> 8) & 0xFF;
    bmp[26] = 1;   // Planes
    bmp[28] = 24;  // Bits per pixel
    
    // Pixel data (BMP is bottom-up)
    for (int y = 0; y < out_size; y++) {
        int src_y = (out_size - 1 - y) / scale;  // Flip vertically
        for (int x = 0; x < out_size; x++) {
            int src_x = x / scale;
            uint8_t val = debug_image[src_y * INPUT_SIZE + src_x];
            int idx = 54 + y * row_bytes + x * 3;
            bmp[idx + 0] = val;  // B
            bmp[idx + 1] = val;  // G
            bmp[idx + 2] = val;  // R
        }
    }
    
    httpd_resp_set_type(req, "image/bmp");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache");
    httpd_resp_send(req, (const char*)bmp, file_size);
    free(bmp);
    
    return ESP_OK;
}

// Flash control handler
static bool flash_state = false;
static esp_err_t flash_handler(httpd_req_t *req) {
    char query[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[4];
        if (httpd_query_key_value(query, "on", val, sizeof(val)) == ESP_OK) {
            flash_state = (atoi(val) == 1);
            digitalWrite(LED_GPIO_NUM, flash_state ? HIGH : LOW);
            Serial.printf("Flash: %s\n", flash_state ? "ON" : "OFF");
        }
    }
    httpd_resp_set_type(req, "text/plain");
    return httpd_resp_send(req, flash_state ? "ON" : "OFF", -1);
}

// Reset handler - flushes camera buffers and clears predictions
// FIX: Don't set current_model_id=-1, keep model but clear predictions with -1 (not 0)
static esp_err_t reset_handler(httpd_req_t *req) {
    Serial.println("=== RESET REQUESTED ===");
    
    // 1. Flush camera buffers (get and return twice to clear any stuck frames)
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb) {
        esp_camera_fb_return(fb);
        Serial.println("Flushed frame 1");
    }
    fb = esp_camera_fb_get();
    if (fb) {
        esp_camera_fb_return(fb);
        Serial.println("Flushed frame 2");
    }
    
    // 2. Clear prediction arrays with INVALID values (not 0!)
    // Setting to -1 indicates "no prediction yet" instead of "predicts 0"
    for (int i = 0; i < 5; i++) {
        last_predictions[i] = -1;  // -1 = no prediction (not 0!)
        last_confidences[i] = -1.0f;  // -1 = invalid
        last_inference_time[i] = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            last_probs[i][j] = 0.0f;
        }
    }
    
    // 3. Clear debug image buffer (fill with gray, not black)
    memset(debug_image, 128, sizeof(debug_image));  // Gray = waiting state
    
    // 4. DON'T reset current_model_id - this causes interpreter issues
    // Keep the current model loaded, just clear predictions
    // current_model_id = -1;  // REMOVED - was causing re-init problems
    
    // 5. Re-init camera sensor settings for fresh capture
    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        Serial.println("Camera sensor reset");
    }
    
    Serial.println("Reset complete - ready for fresh prediction");
    
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", -1);
}

// Threshold control handler (ported from Q2)
static esp_err_t threshold_handler(httpd_req_t *req) {
    char query[64];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
        char val[8];
        if (httpd_query_key_value(query, "value", val, sizeof(val)) == ESP_OK) {
            g_threshold = constrain(atoi(val), 0, 255);
            Serial.printf("Threshold set to: %d\n", g_threshold);
        }
        if (httpd_query_key_value(query, "enable", val, sizeof(val)) == ESP_OK) {
            g_use_threshold = (val[0] == '1');
            Serial.printf("Threshold %s\n", g_use_threshold ? "ON" : "OFF");
        }
    }
    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, "OK", -1);
}

void startServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.stack_size = 16384;
    
    httpd_uri_t index_uri = { "/", HTTP_GET, index_handler, NULL };
    httpd_uri_t predict_uri = { "/predict", HTTP_GET, predict_handler, NULL };
    httpd_uri_t snapshot_uri = { "/snapshot", HTTP_GET, snapshot_handler, NULL };
    httpd_uri_t flash_uri = { "/flash", HTTP_GET, flash_handler, NULL };
    httpd_uri_t debug_uri = { "/debug_input", HTTP_GET, debug_handler, NULL };
    httpd_uri_t reset_uri = { "/reset", HTTP_GET, reset_handler, NULL };
    httpd_uri_t threshold_uri = { "/threshold", HTTP_GET, threshold_handler, NULL };
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &predict_uri);
        httpd_register_uri_handler(camera_httpd, &snapshot_uri);
        httpd_register_uri_handler(camera_httpd, &flash_uri);
        httpd_register_uri_handler(camera_httpd, &debug_uri);
        httpd_register_uri_handler(camera_httpd, &reset_uri);
        httpd_register_uri_handler(camera_httpd, &threshold_uri);
        Serial.println("Web server started");
    }
}

// ==================== SETUP & LOOP ====================
void setup() {
    // 1. Init Serial FIRST and WAIT
    Serial.begin(115200);
    delay(3000); // Wait 3 seconds to let monitor connect
    
    Serial.println("\n\n==============================");
    Serial.println("Q4: Multi-Model Digit Recognition");
    Serial.println("==============================\n");

    // 2. Disable Brownout
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    
    // 3. Check PSRAM
    if (psramFound()) {
        Serial.printf("PSRAM Found: %d KB free\n", ESP.getFreePsram() / 1024);
    } else {
        Serial.println("WARNING: PSRAM NOT FOUND! App will likely crash.");
    }
    
    // 4. Allocate buffers
    // Use heap_caps_malloc to force PSRAM
    tensor_arena = (uint8_t*)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        Serial.println("ERROR: Failed to allocate tensor arena!");
        while(1) delay(1000);
    }
    
    preprocess_gray = (uint8_t*)heap_caps_malloc(WORK_SIZE * WORK_SIZE, MALLOC_CAP_SPIRAM);
    preprocess_bin = (uint8_t*)heap_caps_malloc(WORK_SIZE * WORK_SIZE, MALLOC_CAP_SPIRAM);
    preprocess_dilated = (uint8_t*)heap_caps_malloc(WORK_SIZE * WORK_SIZE, MALLOC_CAP_SPIRAM);
    
    if (!preprocess_gray || !preprocess_bin || !preprocess_dilated) {
        Serial.println("ERROR: Failed to allocate preprocess buffers!");
        while(1) delay(1000);
    }
    Serial.println("Memory allocated successfully.");

    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);
    
    // 5. Init Camera with QQVGA (160x120) for stability
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
    config.xclk_freq_hz = 10000000;  // Lower clock for GRAYSCALE stability
    config.pixel_format = PIXFORMAT_GRAYSCALE;  // GRAYSCALE! Much simpler
    config.frame_size = FRAMESIZE_QQVGA;  // 160x120
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        while(1) delay(1000);
    }
    Serial.println("Camera initialized (160x120)");
    
    // 6. Load default model
    if (!initTFLite(MODEL_SQUEEZENET)) {
        Serial.println("TFLite init failed!");
        while(1) delay(1000);
    }
    
    // 7. WiFi
    #if USE_AP_MODE
        WiFi.mode(WIFI_AP);
        // ... (rest of AP config)
        WiFi.softAP(ap_ssid, ap_password, 1, false, 4);
        Serial.println("\n========== AP MODE ==========");
        Serial.printf("IP: %s\n", WiFi.softAPIP().toString().c_str());
    #else
        WiFi.mode(WIFI_STA);
        WiFi.begin(sta_ssid, sta_password);
        int retry = 0;
        while (WiFi.status() != WL_CONNECTED && retry < 30) {
            delay(500);
            Serial.print(".");
            retry++;
        }
        if (WiFi.status() == WL_CONNECTED) {
            Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.println("\nWifi Connect Failed. Rebooting...");
            ESP.restart();
        }
    #endif
    
    startServer();
    Serial.println("\nSystem ready! Open IP in browser.");
}

void loop() {
    delay(100);
}
