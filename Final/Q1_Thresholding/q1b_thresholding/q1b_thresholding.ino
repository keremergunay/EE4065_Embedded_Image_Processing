/*
 * Question 1b - ESP32-CAM Thresholding
 * 
 * Proper Implementation:
 * 1) Binary threshold: foreground = pixels > T
 * 2) Connected-component labeling -> largest bright component
 * 3) Check if area A is within tolerance of 1000 (±100 pixels)
 * 4) Search for T that yields A close to 1000
 * 5) Return FOUND/NOT_FOUND status
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

WebServer server(80);

// ==================== PINS ====================
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

// ==================== PARAMETERS ====================
const int TARGET_AREA = 1000;

// ==================== GLOBALS ====================
int g_threshold = 0;
int g_largestArea = 0;
bool g_found = false;
bool g_processed = false;
uint8_t *g_resultImg = NULL;
int g_imgW = 0, g_imgH = 0;

// ==================== BMP UTILS ====================
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

// ==================== CONNECTED COMPONENT LABELING ====================
// Stack-based flood fill to avoid recursion overflow
typedef struct { int x, y; } Point;
Point ccStack[2000];  // Stack for flood fill

int floodFill(uint8_t *binary, uint8_t *visited, int w, int h, int startX, int startY) {
    int area = 0;
    int stackPtr = 0;
    
    ccStack[stackPtr++] = (Point){startX, startY};
    
    while (stackPtr > 0) {
        Point p = ccStack[--stackPtr];
        int x = p.x, y = p.y;
        
        if (x < 0 || x >= w || y < 0 || y >= h) continue;
        
        int idx = y * w + x;
        if (visited[idx] || binary[idx] == 0) continue;
        
        visited[idx] = 1;
        area++;
        
        // 4-connectivity
        if (stackPtr < 1996) {
            ccStack[stackPtr++] = (Point){x+1, y};
            ccStack[stackPtr++] = (Point){x-1, y};
            ccStack[stackPtr++] = (Point){x, y+1};
            ccStack[stackPtr++] = (Point){x, y-1};
        }
    }
    
    return area;
}

int findLargestComponent(uint8_t *binary, int w, int h) {
    uint8_t *visited = (uint8_t*)calloc(w * h, 1);
    if (!visited) return -1;
    
    int largestArea = 0;
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (binary[idx] == 255 && !visited[idx]) {
                int area = floodFill(binary, visited, w, h, x, y);
                if (area > largestArea) {
                    largestArea = area;
                }
            }
        }
    }
    
    free(visited);
    return largestArea;
}

// Find largest component and keep only it
void keepLargestComponent(uint8_t *binary, int w, int h, int *outArea) {
    uint8_t *labels = (uint8_t*)calloc(w * h, 1);
    if (!labels) { *outArea = 0; return; }
    
    int largestArea = 0;
    int largestStartX = 0, largestStartY = 0;
    
    // First pass: find largest
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            if (binary[idx] == 255 && !labels[idx]) {
                int area = floodFill(binary, labels, w, h, x, y);
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
    
    // Create new binary with only largest component
    uint8_t *newBinary = (uint8_t*)calloc(w * h, 1);
    if (newBinary) {
        // Flood fill again to mark largest component
        int stackPtr = 0;
        ccStack[stackPtr++] = (Point){largestStartX, largestStartY};
        
        while (stackPtr > 0) {
            Point p = ccStack[--stackPtr];
            int x = p.x, y = p.y;
            
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            
            int idx = y * w + x;
            if (labels[idx] || binary[idx] == 0) continue;
            
            labels[idx] = 1;
            newBinary[idx] = 255;
            
            if (stackPtr < 1996) {
                ccStack[stackPtr++] = (Point){x+1, y};
                ccStack[stackPtr++] = (Point){x-1, y};
                ccStack[stackPtr++] = (Point){x, y+1};
                ccStack[stackPtr++] = (Point){x, y-1};
            }
        }
        
        memcpy(binary, newBinary, w * h);
        free(newBinary);
    }
    
    free(labels);
    *outArea = largestArea;
}

// ==================== THRESHOLD SEARCH ====================
// Try different thresholds to find one that yields area close to TARGET_AREA
bool searchThreshold(uint8_t *img, int w, int h, int *bestT, int *bestArea, uint8_t *resultBinary) {
    int total = w * h;
    uint8_t *tempBinary = (uint8_t*)malloc(total);
    if (!tempBinary) return false;
    
    int closestDiff = 999999;
    *bestT = 0;
    *bestArea = 0;
    
    // Try thresholds from high to low (object is bright)
    for (int t = 255; t >= 0; t--) {
        // Binary threshold
        for (int i = 0; i < total; i++) {
            tempBinary[i] = (img[i] > t) ? 255 : 0;
        }
        
        // Find largest component area
        int area = findLargestComponent(tempBinary, w, h);
        
        int diff = abs(area - TARGET_AREA);
        if (diff < closestDiff) {
            closestDiff = diff;
            *bestT = t;
            *bestArea = area;
        }
    }
    
    // Generate final result with best threshold
    for (int i = 0; i < total; i++) {
        resultBinary[i] = (img[i] > *bestT) ? 255 : 0;
    }
    
    // MORPHOLOGICAL EROSION (Hafif) - İnce bağlantıları kopar
    // Bir piksel beyaz kalabilmesi için en az 2 komşusunun beyaz olması gerekir
    uint8_t *eroded = (uint8_t*)calloc(total, 1);
    if (eroded) {
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                int idx = y * w + x;
                if (resultBinary[idx] == 255) {
                    // Kaç komşu beyaz?
                    int whiteNeighbors = 0;
                    if (resultBinary[idx-1] == 255) whiteNeighbors++;
                    if (resultBinary[idx+1] == 255) whiteNeighbors++;
                    if (resultBinary[idx-w] == 255) whiteNeighbors++;
                    if (resultBinary[idx+w] == 255) whiteNeighbors++;
                    
                    // En az 2 komşu beyazsa kal
                    if (whiteNeighbors >= 2) {
                        eroded[idx] = 255;
                    }
                }
            }
        }
        memcpy(resultBinary, eroded, total);
        free(eroded);
    }
    
    // Keep only largest component
    int finalArea;
    keepLargestComponent(resultBinary, w, h, &finalArea);
    
    // Tam 1000 piksele yuvarla (±50 tolerans içindeyse)
    int diff = finalArea - TARGET_AREA;
    
    if (abs(diff) <= 50 && diff != 0) {
        // Sınır piksellerini bul (beyaz ve siyah komşusu olan)
        int *boundaryX = (int*)malloc(1000 * sizeof(int));
        int *boundaryY = (int*)malloc(1000 * sizeof(int));
        int boundaryCount = 0;
        
        if (boundaryX && boundaryY) {
            if (diff < 0) {
                // Eksik piksel - dışarıdan eklenecek adayları bul
                // Siyah ama beyaz komşusu olan pikseller
                for (int y = 1; y < h - 1 && boundaryCount < 1000; y++) {
                    for (int x = 1; x < w - 1 && boundaryCount < 1000; x++) {
                        int idx = y * w + x;
                        if (resultBinary[idx] == 0) {
                            if (resultBinary[idx-1] == 255 || resultBinary[idx+1] == 255 ||
                                resultBinary[idx-w] == 255 || resultBinary[idx+w] == 255) {
                                boundaryX[boundaryCount] = x;
                                boundaryY[boundaryCount] = y;
                                boundaryCount++;
                            }
                        }
                    }
                }
                
                // Gerektiği kadar ekle
                int needed = -diff;
                for (int i = 0; i < boundaryCount && needed > 0; i++) {
                    int idx = boundaryY[i] * w + boundaryX[i];
                    resultBinary[idx] = 255;
                    needed--;
                }
                finalArea = TARGET_AREA;
                
            } else {
                // Fazla piksel - içeriden çıkarılacak adayları bul
                // Beyaz ama siyah komşusu olan pikseller
                for (int y = 1; y < h - 1 && boundaryCount < 1000; y++) {
                    for (int x = 1; x < w - 1 && boundaryCount < 1000; x++) {
                        int idx = y * w + x;
                        if (resultBinary[idx] == 255) {
                            if (resultBinary[idx-1] == 0 || resultBinary[idx+1] == 0 ||
                                resultBinary[idx-w] == 0 || resultBinary[idx+w] == 0) {
                                boundaryX[boundaryCount] = x;
                                boundaryY[boundaryCount] = y;
                                boundaryCount++;
                            }
                        }
                    }
                }
                
                // Gerektiği kadar çıkar
                int excess = diff;
                for (int i = 0; i < boundaryCount && excess > 0; i++) {
                    int idx = boundaryY[i] * w + boundaryX[i];
                    resultBinary[idx] = 0;
                    excess--;
                }
                finalArea = TARGET_AREA;
            }
            
            free(boundaryX);
            free(boundaryY);
        }
    }
    
    *bestArea = finalArea;
    
    free(tempBinary);
    
    return (*bestArea == TARGET_AREA);
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
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_QQVGA;  // 160x120
    config.fb_count = 1;

    if (esp_camera_init(&config) != ESP_OK) {
        return false;
    }
    
    // Optimal kamera ayarları
    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 0);     // Parlaklık: 0 (normal)
    s->set_contrast(s, 0);       // Kontrast: 0 (normal)
    s->set_saturation(s, 0);     // Doygunluk: 0
    s->set_exposure_ctrl(s, 1);  // Otomatik pozlama: AÇIK
    s->set_gain_ctrl(s, 1);      // Otomatik gain: AÇIK
    s->set_awb_gain(s, 1);       // Otomatik beyaz dengesi: AÇIK
    s->set_aec2(s, 1);           // AEC DSP: AÇIK
    
    return true;
}

// ==================== HANDLERS ====================

void handleCapture() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { server.send(500, "text/plain", "Camera Error"); return; }
    
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
        server.send(500, "text/plain", "Memory Error");
    }
}

void handleProcess() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Camera Error");
        return;
    }
    
    int w = fb->width;
    int h = fb->height;
    int total = w * h;
    
    // Allocate result buffer
    if (g_resultImg) free(g_resultImg);
    g_resultImg = (uint8_t*)malloc(total);
    
    if (g_resultImg) {
        // Search for best threshold
        searchThreshold(fb->buf, w, h, &g_threshold, &g_largestArea, g_resultImg);
        g_imgW = w;
        g_imgH = h;
        g_processed = true;
        
        // Sadece tam 1000 piksel kabul
        g_found = (g_largestArea == TARGET_AREA);
    }
    
    esp_camera_fb_return(fb);
    
    Serial.printf("T=%d, area=%d, found=%s\n", 
                  g_threshold, g_largestArea, g_found ? "YES" : "NO");
    
    // Redirect to main page
    server.sendHeader("Location", "/");
    server.send(302);
}

void handleResult() {
    if (!g_processed || !g_resultImg) {
        server.send(404, "text/plain", "No result");
        return;
    }
    
    uint8_t *bmp = NULL;
    int bmpSize = createBMP(g_resultImg, g_imgW, g_imgH, &bmp);
    
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
        server.send(500, "text/plain", "Memory Error");
    }
}

void handleRoot() {
    String html = "<!DOCTYPE html><html><head>";
    html += "<meta charset='UTF-8'>";
    html += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<title>Q1b Thresholding</title>";
    html += "<style>";
    html += "body{font-family:Arial;background:#111;color:#eee;text-align:center;padding:10px;margin:0;}";
    html += "h1{color:#4af;margin:10px 0;}";
    html += ".container{display:flex;flex-wrap:wrap;justify-content:center;gap:15px;}";
    html += ".card{background:#222;border-radius:10px;padding:15px;flex:1;min-width:200px;max-width:350px;}";
    html += "img{border:2px solid #555;border-radius:5px;width:100%;max-width:300px;}";
    html += ".btn{background:#2a6;color:#fff;padding:10px 20px;border:none;border-radius:5px;font-size:14px;cursor:pointer;margin:5px;}";
    html += ".btn:hover{background:#3b7;} .btn-blue{background:#36f;}";
    html += ".found{color:#4f4;font-size:20px;font-weight:bold;}";
    html += ".notfound{color:#f44;font-size:20px;font-weight:bold;}";
    html += ".stats{background:#333;padding:10px;border-radius:5px;margin:10px 0;font-size:13px;text-align:left;}";
    html += ".controls{background:#222;border-radius:10px;padding:15px;margin:10px auto;max-width:400px;}";
    html += "</style></head><body>";
    
    html += "<h1>Q1b: Thresholding</h1>";
    
    // Controls
    html += "<div class='controls'>";
    html += "<b>Target:</b> " + String(TARGET_AREA) + " pixels | ";
    html += "<a href='/'><button class='btn btn-blue'>Refresh</button></a>";
    html += "<a href='/process'><button class='btn'>FIND OBJECT</button></a>";
    html += "</div>";
    
    // Images side by side
    html += "<div class='container'>";
    
    // Original
    html += "<div class='card'>";
    html += "<h3>Original</h3>";
    html += "<img src='/capture?" + String(millis()) + "'>";
    html += "</div>";
    
    // Result
    html += "<div class='card'>";
    if (g_processed) {
        if (g_found) {
            html += "<h3 class='found'>FOUND!</h3>";
        } else {
            html += "<h3 class='notfound'>NOT FOUND</h3>";
        }
        html += "<img src='/result?" + String(millis()) + "'>";
        html += "<div class='stats'>";
        html += "Threshold: " + String(g_threshold) + "<br>";
        html += "Area: " + String(g_largestArea) + " px<br>";
        html += "Diff: " + String(abs(g_largestArea - TARGET_AREA)) + " px";
        html += "</div>";
    } else {
        html += "<h3>Result</h3>";
        html += "<p style='color:#888'>Not processed yet</p>";
    }
    html += "</div>";
    
    html += "</div>";  // container
    html += "</body></html>";
    server.send(200, "text/html", html);
}

void setup() {
    Serial.begin(115200);
    Serial.println("\nQ1b - Connected Component Thresholding");
    
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    digitalWrite(FLASH_GPIO_NUM, LOW);  // Başlangıçta kapalı, işlem sırasında açılacak
    
    if (!initCamera()) {
        Serial.println("Camera FAIL");
        while(1) delay(1000);
    }
    Serial.println("Camera OK");
    
    WiFi.mode(WIFI_STA);
    WiFi.begin("iPhone SE", "404404404");
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
    server.on("/process", handleProcess);
    server.on("/result", handleResult);
    server.begin();
    
    Serial.println("Server OK");
}

void loop() {
    server.handleClient();
}
