/*
 * Question 3 - Image Resampling with Bilinear Interpolation
 * Supports non-integer scale factors (1.5, 2/3, etc.)
 * 
 * Board: AI Thinker ESP32-CAM
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ==================== WiFi CONFIG ====================
const char* ssid = "iPhone SE";
const char* password = "404404404";

WebServer server(80);

// ==================== CAMERA PINS ====================
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

// ==================== OUTPUT BUFFER ====================
#define MAX_OUT_SIZE 45000  // 240x180 for x1.5 upsample
static uint8_t output_buffer[MAX_OUT_SIZE];

// ==================== RESULT STORAGE ====================
struct SamplingResult {
    int orig_w, orig_h;
    int new_w, new_h;
    float scale;
    String method;
    bool success;
} lastResult;

// ==================== BILINEAR INTERPOLATION API ====================

/*
 * API A: Allocate and return resampled image
 * Returns: pointer to allocated buffer (caller must free)
 * Sets: dst_w, dst_h with output dimensions
 */
uint8_t* resample_bilinear_alloc(const uint8_t *src, int src_w, int src_h, 
                                  float scale, int *dst_w, int *dst_h) {
    *dst_w = (int)round(src_w * scale);
    *dst_h = (int)round(src_h * scale);
    
    if (*dst_w < 1) *dst_w = 1;
    if (*dst_h < 1) *dst_h = 1;
    
    uint8_t *dst = (uint8_t*)malloc((*dst_w) * (*dst_h));
    if (!dst) return NULL;
    
    // Bilinear interpolation
    for (int y_out = 0; y_out < *dst_h; y_out++) {
        for (int x_out = 0; x_out < *dst_w; x_out++) {
            // Map output pixel to source coordinates
            float x_in = x_out / scale;
            float y_in = y_out / scale;
            
            // Get integer coordinates
            int x0 = (int)floor(x_in);
            int y0 = (int)floor(y_in);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Clamp to valid range
            if (x0 < 0) x0 = 0;
            if (y0 < 0) y0 = 0;
            if (x1 >= src_w) x1 = src_w - 1;
            if (y1 >= src_h) y1 = src_h - 1;
            if (x0 >= src_w) x0 = src_w - 1;
            if (y0 >= src_h) y0 = src_h - 1;
            
            // Fractional parts (weights)
            float dx = x_in - floor(x_in);
            float dy = y_in - floor(y_in);
            
            // Get 4 neighbor pixels
            uint8_t I00 = src[y0 * src_w + x0];
            uint8_t I10 = src[y0 * src_w + x1];
            uint8_t I01 = src[y1 * src_w + x0];
            uint8_t I11 = src[y1 * src_w + x1];
            
            // Bilinear interpolation formula
            float p = (1-dx)*(1-dy)*I00 + dx*(1-dy)*I10 + 
                      (1-dx)*dy*I01 + dx*dy*I11;
            
            // Round and clamp to [0, 255]
            int val = (int)round(p);
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            
            dst[y_out * (*dst_w) + x_out] = (uint8_t)val;
        }
    }
    
    return dst;
}

/*
 * API B: Write into preallocated buffer
 * Returns: true on success, false on error
 * Requires: dst buffer with at least dst_w * dst_h bytes
 */
bool resample_bilinear_into(const uint8_t *src, int src_w, int src_h,
                            float scale, uint8_t *dst, int *dst_w, int *dst_h) {
    *dst_w = (int)round(src_w * scale);
    *dst_h = (int)round(src_h * scale);
    
    if (*dst_w < 1) *dst_w = 1;
    if (*dst_h < 1) *dst_h = 1;
    
    if (!dst) return false;
    
    // Bilinear interpolation (same as above)
    for (int y_out = 0; y_out < *dst_h; y_out++) {
        for (int x_out = 0; x_out < *dst_w; x_out++) {
            float x_in = x_out / scale;
            float y_in = y_out / scale;
            
            int x0 = (int)floor(x_in);
            int y0 = (int)floor(y_in);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            // Clamp
            x0 = constrain(x0, 0, src_w - 1);
            y0 = constrain(y0, 0, src_h - 1);
            x1 = constrain(x1, 0, src_w - 1);
            y1 = constrain(y1, 0, src_h - 1);
            
            float dx = x_in - floor(x_in);
            float dy = y_in - floor(y_in);
            
            uint8_t I00 = src[y0 * src_w + x0];
            uint8_t I10 = src[y0 * src_w + x1];
            uint8_t I01 = src[y1 * src_w + x0];
            uint8_t I11 = src[y1 * src_w + x1];
            
            float p = (1-dx)*(1-dy)*I00 + dx*(1-dy)*I10 + 
                      (1-dx)*dy*I01 + dx*dy*I11;
            
            dst[y_out * (*dst_w) + x_out] = (uint8_t)constrain((int)round(p), 0, 255);
        }
    }
    
    return true;
}

// ==================== BMP CREATION ====================
int createBMP(uint8_t *gray, int w, int h, uint8_t **outBmp) {
    int rowSize = (w + 3) & ~3;
    int paletteSize = 256 * 4;
    int pixelOffset = 54 + paletteSize;
    int fileSize = pixelOffset + rowSize * h;
    
    uint8_t *bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) return 0;
    
    memset(bmp, 0, fileSize);
    
    bmp[0] = 'B'; bmp[1] = 'M';
    bmp[2] = fileSize & 0xFF; bmp[3] = (fileSize >> 8) & 0xFF;
    bmp[4] = (fileSize >> 16) & 0xFF; bmp[5] = (fileSize >> 24) & 0xFF;
    bmp[10] = pixelOffset & 0xFF; bmp[11] = (pixelOffset >> 8) & 0xFF;
    
    bmp[14] = 40;
    bmp[18] = w & 0xFF; bmp[19] = (w >> 8) & 0xFF;
    bmp[22] = h & 0xFF; bmp[23] = (h >> 8) & 0xFF;
    bmp[26] = 1; bmp[28] = 8;
    
    for (int i = 0; i < 256; i++) {
        bmp[54 + i*4] = i; bmp[54 + i*4 + 1] = i; bmp[54 + i*4 + 2] = i;
    }
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bmp[pixelOffset + (h - 1 - y) * rowSize + x] = gray[y * w + x];
        }
    }
    
    *outBmp = bmp;
    return fileSize;
}

// ==================== CAMERA INIT ====================
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
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_QQVGA;  // 160x120
    config.fb_count = 1;

    return (esp_camera_init(&config) == ESP_OK);
}

// ==================== WEB HANDLERS ====================

void handleRoot() {
    String html = "<!DOCTYPE html><html><head>";
    html += "<meta charset='UTF-8'>";
    html += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<title>Q3 Bilinear Resampling</title>";
    html += "<style>";
    html += "body{font-family:Arial;background:#111;color:#eee;text-align:center;padding:10px;margin:0;}";
    html += "h1{color:#4af;margin:10px 0;}";
    html += ".container{display:flex;flex-wrap:wrap;justify-content:center;gap:15px;max-width:1200px;margin:0 auto;}";
    html += ".card{background:#222;border-radius:10px;padding:15px;flex:1;min-width:250px;max-width:400px;}";
    html += "img{border:2px solid #555;border-radius:5px;width:100%;max-width:320px;}";
    html += ".btn{background:#2a6;color:#fff;padding:10px 16px;border:none;border-radius:5px;font-size:13px;cursor:pointer;margin:3px;text-decoration:none;display:inline-block;}";
    html += ".btn:hover{background:#3b7;} .btn-down{background:#c55;}";
    html += ".stats{background:#333;padding:8px;border-radius:5px;margin:8px 0;font-size:12px;}";
    html += ".controls{background:#222;border-radius:10px;padding:10px;margin:10px auto;max-width:600px;}";
    html += ".label{color:#888;font-size:11px;}";
    html += "</style></head><body>";
    
    html += "<h1>Q3: Bilinear Resampling</h1>";
    
    // Controls
    html += "<div class='controls'>";
    html += "<a href='/'><button class='btn'>Refresh</button></a> ";
    html += "<b>Upsample:</b> ";
    html += "<a href='/resample?s=1.5' class='btn'>x1.5</a>";
    html += " <b>Downsample:</b> ";
    html += "<a href='/resample?s=0.5' class='btn btn-down'>x0.5</a>";
    html += "<a href='/resample?s=0.667' class='btn btn-down'>x2/3</a>";
    html += "</div>";
    
    // Images side by side
    html += "<div class='container'>";
    
    // Original
    html += "<div class='card'>";
    html += "<h3>Original (160x120)</h3>";
    html += "<img src='/capture?" + String(millis()) + "'>";
    html += "</div>";
    
    // Result
    html += "<div class='card'>";
    if (lastResult.success) {
        html += "<h3>Resampled (" + String(lastResult.new_w) + "x" + String(lastResult.new_h) + ")</h3>";
        html += "<img src='/result?" + String(millis()) + "'>";
        html += "<div class='stats'>";
        html += "Scale: x" + String(lastResult.scale, 2) + "<br>";
        html += lastResult.method + "<br>";
        html += String(lastResult.orig_w) + "x" + String(lastResult.orig_h) + " → " + 
                String(lastResult.new_w) + "x" + String(lastResult.new_h);
        html += "</div>";
    } else {
        html += "<h3>Result</h3>";
        html += "<p style='color:#888'>Click a scale button</p>";
    }
    html += "</div>";
    
    html += "</div>";  // container
    html += "</body></html>";
    server.send(200, "text/html", html);
}

void handleCapture() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Capture failed"); return; }
    
    uint8_t *bmp = NULL;
    int bmpSize = createBMP(fb->buf, fb->width, fb->height, &bmp);
    esp_camera_fb_return(fb);
    
    if (bmp && bmpSize > 0) {
        WiFiClient client = server.client();
        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: image/bmp");
        client.println("Content-Length: " + String(bmpSize));
        client.println("Connection: close");
        client.println();
        client.write(bmp, bmpSize);
        free(bmp);
    } else {
        server.send(500, "text/plain", "BMP Error");
    }
}

void handleResample() {
    float scale = 1.5;
    if (server.hasArg("s")) {
        scale = server.arg("s").toFloat();
    }
    if (scale <= 0 || scale > 3.0) scale = 1.5;
    
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Capture failed");
        return;
    }
    
    int out_w, out_h;
    
    // Check if output fits in buffer
    int expected_size = (int)(fb->width * scale * fb->height * scale);
    if (expected_size > MAX_OUT_SIZE) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "Output too large");
        return;
    }
    
    // Use API B (into preallocated buffer)
    bool ok = resample_bilinear_into(fb->buf, fb->width, fb->height, 
                                      scale, output_buffer, &out_w, &out_h);
    
    lastResult.orig_w = fb->width;
    lastResult.orig_h = fb->height;
    lastResult.new_w = out_w;
    lastResult.new_h = out_h;
    lastResult.scale = scale;
    lastResult.method = (scale >= 1.0) ? "Bilinear Upsample" : "Bilinear Downsample";
    lastResult.success = ok;
    
    esp_camera_fb_return(fb);
    
    if (!ok) {
        server.send(500, "text/plain", "Resample failed");
        return;
    }
    
    Serial.printf("Resampled: %dx%d -> %dx%d (x%.2f)\n", 
                  lastResult.orig_w, lastResult.orig_h, out_w, out_h, scale);
    
    // Redirect to main page
    server.sendHeader("Location", "/");
    server.send(302);
}

void handleResult() {
    if (!lastResult.success) {
        server.send(404, "text/plain", "No result");
        return;
    }
    
    uint8_t *bmp = NULL;
    int bmpSize = createBMP(output_buffer, lastResult.new_w, lastResult.new_h, &bmp);
    
    if (bmp && bmpSize > 0) {
        WiFiClient client = server.client();
        client.println("HTTP/1.1 200 OK");
        client.println("Content-Type: image/bmp");
        client.println("Content-Length: " + String(bmpSize));
        client.println("Connection: close");
        client.println();
        client.write(bmp, bmpSize);
        free(bmp);
    } else {
        server.send(500, "text/plain", "BMP Error");
    }
}

// ==================== SETUP & LOOP ====================

void setup() {
    Serial.begin(115200);
    Serial.println("\nQ3: Bilinear Resampling - ESP32-CAM");
    
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, LOW);
    
    if (!initCamera()) {
        Serial.println("Camera FAIL");
        while(1) delay(1000);
    }
    Serial.println("Camera OK");
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    Serial.print("IP: http://");
    Serial.println(WiFi.localIP());
    
    server.on("/", handleRoot);
    server.on("/capture", handleCapture);
    server.on("/resample", handleResample);
    server.on("/result", handleResult);
    server.begin();
    
    Serial.println("Server ready!");
}

void loop() {
    server.handleClient();
}
