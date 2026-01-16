/*
 * EE4065 - Final Project - Question 5a
 * FOMO Digit Detection on ESP32-CAM
 * SIMPLIFIED DEBUG VERSION
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "model_data.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// WiFi
const char* ssid = "iPhone SE";
const char* password = "404404404";

// Model
#define INPUT_SIZE 128
#define GRID_SIZE 12
#define NUM_CLASSES 11
#define THRESHOLD 0.4

// AI-Thinker ESP32-CAM pins
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

WebServer server(80);

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::MicroErrorReporter error_reporter;

constexpr int kArenaSize = 150 * 1024;
uint8_t* arena = nullptr;

struct Det { int digit, x, y; float conf; };
Det dets[10];
int numDets = 0;
unsigned long infTime = 0;

// Debug counters
uint32_t frameNum = 0;
uint32_t lastPixelSum = 0;

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
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;  // Single buffer for simplicity
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 2);
    
    Serial.println("Camera OK");
    return true;
}

bool initModel() {
    arena = (uint8_t*)heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!arena) arena = (uint8_t*)malloc(kArenaSize);
    if (!arena) { Serial.println("Arena failed!"); return false; }
    
    model = tflite::GetModel(model_data);
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interp(model, resolver, arena, kArenaSize, &error_reporter);
    interpreter = &interp;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Allocate failed!");
        return false;
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.printf("Model OK. Input type: %d\n", input->type);
    return true;
}

void doInference(uint8_t* img) {
    unsigned long t0 = millis();
    
    // Calculate average brightness for adaptive threshold
    uint32_t sum = 0;
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
        sum += img[i];
    }
    uint8_t avg = sum / (INPUT_SIZE * INPUT_SIZE);
    uint8_t threshold = avg - 30;  // Pixels darker than avg-30 are "ink"
    
    Serial.printf("Avg brightness: %d, threshold: %d\n", avg, threshold);
    
    // Apply threshold + invert to make MNIST-like
    // MNIST: white digits (255) on black background (0)
    // Camera: dark ink on bright paper
    // After processing: bright paper -> 0 (black), dark ink -> 255 (white)
    
    if (input->type == kTfLiteUInt8) {
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
            // If pixel is darker than threshold -> it's ink -> make it white (255)
            // Otherwise -> it's paper -> make it black (0)
            input->data.uint8[i] = (img[i] < threshold) ? 255 : 0;
        }
    } else {
        for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
            input->data.f[i] = (img[i] < threshold) ? 1.0f : 0.0f;
        }
    }
    
    interpreter->Invoke();
    infTime = millis() - t0;
    
    // Find detections
    numDets = 0;
    float scale = (output->type == kTfLiteUInt8) ? output->params.scale : 1.0f;
    int zp = (output->type == kTfLiteUInt8) ? output->params.zero_point : 0;
    
    for (int gy = 0; gy < GRID_SIZE; gy++) {
        for (int gx = 0; gx < GRID_SIZE; gx++) {
            int bestC = 0;
            float bestConf = 0;
            
            for (int c = 0; c < NUM_CLASSES; c++) {
                int idx = (gy * GRID_SIZE + gx) * NUM_CLASSES + c;
                float conf;
                if (output->type == kTfLiteUInt8) {
                    conf = (output->data.uint8[idx] - zp) * scale;
                } else {
                    conf = output->data.f[idx];
                }
                if (conf > bestConf) {
                    bestConf = conf;
                    bestC = c;
                }
            }
            
            if (bestC > 0 && bestConf > THRESHOLD && numDets < 10) {
                dets[numDets].digit = bestC - 1;
                dets[numDets].x = gx * 8 + 4;
                dets[numDets].y = gy * 8 + 4;
                dets[numDets].conf = bestConf;
                numDets++;
            }
        }
    }
    
    Serial.printf("Inference: %dms, Detections: %d\n", infTime, numDets);
}

// Modern Blue UI
const char* html = R"HTML(
<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FOMO Digit Detection</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:'Segoe UI',system-ui,sans-serif;
  background:linear-gradient(135deg,#0c1929 0%,#1a3a5c 50%,#0d2137 100%);
  min-height:100vh;
  color:#e8f4fc;
  overflow-x:hidden;
}
.bg-orbs{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0}
.orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:0.3;animation:float 8s infinite ease-in-out}
.orb1{width:400px;height:400px;background:#0ea5e9;top:-100px;left:-100px}
.orb2{width:300px;height:300px;background:#3b82f6;bottom:-50px;right:-50px;animation-delay:-4s}
.orb3{width:200px;height:200px;background:#06b6d4;top:50%;left:50%;animation-delay:-2s}
@keyframes float{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-30px) scale(1.05)}}

.container{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:30px}
header{text-align:center;margin-bottom:30px}
h1{
  font-size:2.5em;
  font-weight:700;
  background:linear-gradient(90deg,#38bdf8,#818cf8,#38bdf8);
  background-size:200% auto;
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  animation:shimmer 3s linear infinite;
}
@keyframes shimmer{to{background-position:200% center}}
.subtitle{color:#64748b;margin-top:8px;font-size:1.1em}

.main-layout{display:flex;gap:25px;flex-wrap:wrap;justify-content:center}

.card{
  background:rgba(30,58,95,0.6);
  backdrop-filter:blur(20px);
  border-radius:20px;
  border:1px solid rgba(56,189,248,0.2);
  padding:25px;
  box-shadow:0 8px 32px rgba(0,0,0,0.3),inset 0 1px 0 rgba(255,255,255,0.1);
  transition:transform 0.3s,box-shadow 0.3s;
}
.card:hover{transform:translateY(-5px);box-shadow:0 15px 40px rgba(56,189,248,0.15)}

.camera-card{flex:1;min-width:320px;max-width:400px}
.results-card{flex:1;min-width:320px;max-width:450px}

.card-title{
  display:flex;align-items:center;gap:10px;
  font-size:1.2em;font-weight:600;
  color:#38bdf8;margin-bottom:20px;
  padding-bottom:12px;
  border-bottom:1px solid rgba(56,189,248,0.2);
}
.card-title svg{width:24px;height:24px}

.cam-wrapper{
  position:relative;
  background:#0f172a;
  border-radius:16px;
  overflow:hidden;
  border:2px solid rgba(56,189,248,0.3);
}
#cam{width:100%;height:auto;display:block;image-rendering:pixelated}
.cam-overlay{
  position:absolute;top:10px;right:10px;
  background:rgba(0,0,0,0.6);
  padding:5px 12px;border-radius:20px;
  font-size:0.8em;color:#38bdf8;
}
.live-dot{display:inline-block;width:8px;height:8px;background:#22c55e;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

.btn-row{display:flex;gap:12px;margin-top:20px;flex-wrap:wrap;justify-content:center}

.btn{
  position:relative;overflow:hidden;
  padding:14px 28px;
  font-size:1em;font-weight:600;
  border:none;border-radius:12px;
  cursor:pointer;
  transition:all 0.3s;
  display:flex;align-items:center;gap:8px;
}
.btn::before{
  content:'';position:absolute;top:0;left:-100%;
  width:100%;height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.2),transparent);
  transition:left 0.5s;
}
.btn:hover::before{left:100%}

.btn-primary{
  background:linear-gradient(135deg,#0ea5e9,#3b82f6);
  color:#fff;
  box-shadow:0 4px 20px rgba(14,165,233,0.4);
}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 6px 30px rgba(14,165,233,0.5)}

.btn-secondary{
  background:rgba(56,189,248,0.15);
  color:#38bdf8;
  border:1px solid rgba(56,189,248,0.3);
}
.btn-secondary:hover{background:rgba(56,189,248,0.25)}

.stats{
  display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:20px;
}
.stat-box{
  background:rgba(15,23,42,0.6);
  border-radius:12px;padding:15px;text-align:center;
  border:1px solid rgba(56,189,248,0.15);
}
.stat-val{font-size:1.8em;font-weight:700;color:#38bdf8}
.stat-label{font-size:0.8em;color:#64748b;margin-top:4px}

#detList{max-height:300px;overflow-y:auto;padding-right:5px}
.det-item{
  display:flex;align-items:center;gap:15px;
  background:rgba(15,23,42,0.5);
  border-radius:12px;padding:15px;margin:10px 0;
  border-left:4px solid #38bdf8;
  animation:slideIn 0.3s ease-out;
}
@keyframes slideIn{from{opacity:0;transform:translateX(-20px)}to{opacity:1;transform:translateX(0)}}

.det-digit{
  font-size:2.5em;font-weight:700;
  width:60px;height:60px;
  display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#0ea5e9,#6366f1);
  border-radius:12px;color:#fff;
}
.det-info{flex:1}
.det-conf{font-size:1.1em;color:#94a3b8}
.det-pos{font-size:0.85em;color:#64748b;font-family:monospace;margin-top:4px}

.no-det{
  text-align:center;padding:40px;color:#475569;
  background:rgba(15,23,42,0.3);border-radius:12px;
}
.no-det-icon{font-size:3em;margin-bottom:10px;opacity:0.5}

footer{text-align:center;margin-top:30px;color:#475569;font-size:0.9em}
</style>
</head><body>
<div class="bg-orbs"><div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div></div>
<div class="container">
<header>
<h1>FOMO Digit Detection</h1>
<p class="subtitle">EE4065 Final Project - ESP32-CAM + TensorFlow Lite</p>
</header>
<div class="main-layout">
<div class="card camera-card">
<div class="card-title">
<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
Kamera Goruntusu
</div>
<div class="cam-wrapper">
<img id="cam" src="/img?0" width="288" height="288">
<div class="cam-overlay"><span class="live-dot"></span>LIVE</div>
</div>
<div class="btn-row">
<button class="btn btn-primary" onclick="detect()">
<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
Tespit Et
</button>
<button class="btn btn-secondary" onclick="refresh()">
<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
Yenile
</button>
</div>
</div>
<div class="card results-card">
<div class="card-title">
<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
Sonuclar
</div>
<div class="stats">
<div class="stat-box"><div class="stat-val" id="sFrame">0</div><div class="stat-label">Kare</div></div>
<div class="stat-box"><div class="stat-val" id="sTime">-</div><div class="stat-label">ms</div></div>
<div class="stat-box"><div class="stat-val" id="sCount">0</div><div class="stat-label">Tespit</div></div>
</div>
<div id="detList">
<div class="no-det"><div class="no-det-icon">🔍</div>Tespit icin butona basin</div>
</div>
</div>
</div>
<footer>ESP32-CAM • FOMO Model • 96x96 Grayscale Input</footer>
</div>
<script>
var n=0;
function refresh(){n++;document.getElementById("cam").src="/img?"+n;}
function detect(){
  n++;
  document.getElementById("cam").src="/img?"+n;
  fetch("/run").then(r=>r.text()).then(t=>{
    var lines=t.split("\n");
    var frame=0,time=0,count=0,dets=[];
    lines.forEach(function(l){
      if(l.startsWith("Frame:"))frame=parseInt(l.split(":")[1]);
      if(l.startsWith("Inference"))time=parseInt(l.match(/\d+/)[0]);
      if(l.startsWith("Detections:"))count=parseInt(l.split(":")[1]);
      if(l.startsWith("Digit")){
        var m=l.match(/Digit (\d+) at \((\d+),(\d+)\) conf=([\d.]+)/);
        if(m)dets.push({d:m[1],x:m[2],y:m[3],c:parseFloat(m[4])});
      }
    });
    document.getElementById("sFrame").innerText=frame;
    document.getElementById("sTime").innerText=time;
    document.getElementById("sCount").innerText=count;
    var h="";
    if(dets.length==0){
      h='<div class="no-det"><div class="no-det-icon">❌</div>Rakam bulunamadi</div>';
    }else{
      dets.forEach(function(d){
        h+='<div class="det-item">';
        h+='<div class="det-digit">'+d.d+'</div>';
        h+='<div class="det-info">';
        h+='<div class="det-conf">Guven: '+(d.c*100).toFixed(1)+'%</div>';
        h+='<div class="det-pos">Konum: ('+d.x+', '+d.y+')</div>';
        h+='</div></div>';
      });
    }
    document.getElementById("detList").innerHTML=h;
  });
}
setInterval(refresh,2000);
</script>
</body></html>
)HTML";

void handleRoot() {
    server.send(200, "text/html", html);
}

void handleImg() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "No frame");
        return;
    }
    
    frameNum++;
    Serial.printf("Frame %d, len=%d, first4bytes: %02X %02X %02X %02X\n", 
        frameNum, fb->len, fb->buf[0], fb->buf[1], fb->buf[2], fb->buf[3]);
    
    // Convert to simple BMP
    const int w=96, h=96;
    const int hdrSize = 54+256*4;
    const int imgSize = w*h;
    const int fileSize = hdrSize + imgSize;
    
    uint8_t* bmp = (uint8_t*)malloc(fileSize);
    if (!bmp) {
        esp_camera_fb_return(fb);
        server.send(500, "text/plain", "malloc fail");
        return;
    }
    
    memset(bmp, 0, fileSize);
    bmp[0]='B'; bmp[1]='M';
    *(uint32_t*)(bmp+2) = fileSize;
    *(uint32_t*)(bmp+10) = hdrSize;
    *(uint32_t*)(bmp+14) = 40;
    *(int32_t*)(bmp+18) = w;
    *(int32_t*)(bmp+22) = h;
    *(uint16_t*)(bmp+26) = 1;
    *(uint16_t*)(bmp+28) = 8;
    *(uint32_t*)(bmp+34) = imgSize;
    *(uint32_t*)(bmp+46) = 256;
    
    for (int i=0;i<256;i++) {
        bmp[54+i*4+0]=i; bmp[54+i*4+1]=i; bmp[54+i*4+2]=i;
    }
    
    for (int y=0;y<h;y++) {
        for (int x=0;x<w;x++) {
            bmp[hdrSize + (h-1-y)*w + x] = fb->buf[y*w+x];
        }
    }
    
    esp_camera_fb_return(fb);
    
    server.send_P(200, "image/bmp", (const char*)bmp, fileSize);
    free(bmp);
}

void handleRun() {
    // Get fresh frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(200, "text/plain", "ERROR: No camera frame!");
        return;
    }
    
    Serial.printf("\n=== RUN INFERENCE ===\n");
    Serial.printf("Frame size: %d bytes\n", fb->len);
    Serial.printf("First pixels: %d %d %d %d\n", fb->buf[0], fb->buf[1], fb->buf[2], fb->buf[3]);
    Serial.printf("Center pixel: %d\n", fb->buf[48*96+48]);
    
    doInference(fb->buf);
    
    esp_camera_fb_return(fb);
    
    // Build result text
    String result = "Frame: " + String(frameNum) + "\n";
    result += "Inference time: " + String(infTime) + " ms\n";
    result += "Detections: " + String(numDets) + "\n\n";
    
    if (numDets == 0) {
        result += "No digits detected\n";
    } else {
        for (int i = 0; i < numDets; i++) {
            result += "Digit " + String(dets[i].digit);
            result += " at (" + String(dets[i].x) + "," + String(dets[i].y) + ")";
            result += " conf=" + String(dets[i].conf * 100, 1) + "%\n";
        }
    }
    
    server.send(200, "text/plain", result);
}

void setup() {
    Serial.begin(115200);
    Serial.println("\n\n=== FOMO DEBUG ===");
    
    pinMode(FLASH_GPIO_NUM, OUTPUT);
    
    if (!initCamera()) while(1) delay(1000);
    if (!initModel()) while(1) delay(1000);
    
    WiFi.begin(ssid, password);
    Serial.print("WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nIP: " + WiFi.localIP().toString());
    
    server.on("/", handleRoot);
    server.on("/img", handleImg);
    server.on("/run", handleRun);
    server.begin();
    
    Serial.println("Ready!");
}

void loop() {
    server.handleClient();
}
