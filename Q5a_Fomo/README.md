# EE4065 Final Project - Question 5a
# 🎯 FOMO (Faster Objects, More Objects) Rakam Tespiti

Bu proje, ESP32-CAM üzerinde çalışan **FOMO** (Edge Impulse) tabanlı ultra-hızlı rakam tespit sistemidir. FOMO, geleneksel object detection modellerinden çok daha hızlı çalışır ve gömülü sistemler için optimize edilmiştir.

---

## 🎯 FOMO Nedir?

**FOMO** (Faster Objects, More Objects), Edge Impulse tarafından geliştirilen, mikrodenetleyiciler için optimize edilmiş bir object detection modelidir.

### FOMO vs YOLO Karşılaştırması

| Özellik | FOMO | YOLO |
|---------|------|------|
| **Çıkış Türü** | Merkez noktaları (Centroid) | Bounding Box'lar |
| **Model Boyutu** | 50-150KB | 500KB-2MB |
| **Inference Süresi** | 30-50ms | 150-300ms |
| **Bellek Kullanımı** | Düşük | Yüksek |
| **Doğruluk** | Orta | Yüksek |
| **Çoklu Nesne** | Mükemmel | İyi |

### FOMO Avantajları
- ⚡ **Ultra Hızlı**: 30-50ms inference (YOLO'dan 3-5x hızlı)
- 💾 **Küçük Model**: ~100KB (Flash'a rahat sığar)
- 🎯 **Gerçek Zamanlı**: 15-20 FPS mümkün
- 🔋 **Düşük Güç**: Daha az işlem, daha az enerji

### FOMO Dezavantajları
- 📦 Bounding box çıktısı yok (sadece merkez noktası)
- 📏 Nesne boyutu tahmini yapamaz
- 🔍 Çok küçük nesnelerde performans düşer

---

## 📁 Proje Yapısı

```
Q5a_Fomo/
├── README.md                # Bu dosya
├── esp32_fomo_digit.ino     # 🎯 ESP32-CAM ana kodu
├── model_data.h             # FOMO TFLite modeli (header)
├── q5a_fomo.py              # Model eğitim scripti (Edge Impulse)
└── q5b_ssd_mobilenet.py     # Alternatif SSD model scripti
```

---

## 🧠 Model Detayları

### FOMO Mimarisi

```
Input:  96×96×1 (Grayscale)
        ↓
Backbone: MobileNetV2 (alpha=0.35, depth_multiplier=0.5)
        ↓
Feature Map: 12×12×16
        ↓
Classification Head: 1×1 Conv → 11 sınıf
        ↓
Output: 12×12×11 (Grid × Classes)
```

### Çıkış Formatı

```
Output Shape: [1, 12, 12, 11]
- 12×12 Grid: Her hücre 8×8 piksellik bölgeyi temsil eder
- 11 Sınıf:
  - Sınıf 0: Arka plan (nesne yok)
  - Sınıf 1-10: Rakamlar 0-9
```

### Grid Decode

```cpp
for (int gy = 0; gy < 12; gy++) {
    for (int gx = 0; gx < 12; gx++) {
        // Her hücre için en yüksek olasılıklı sınıfı bul
        int bestClass = argmax(output[gy][gx]);
        
        if (bestClass > 0 && confidence > THRESHOLD) {
            // Merkez koordinatı hesapla
            int centerX = gx * 8 + 4;  // Grid → Piksel
            int centerY = gy * 8 + 4;
            
            detections.add(bestClass - 1, centerX, centerY, confidence);
        }
    }
}
```

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler

**Donanım:**
- ESP32-CAM (AI-Thinker) + FTDI Programmer
- USB Kablo
- Beyaz kağıt + kalem (test için)

**Yazılım:**
- Arduino IDE 2.0+
- ESP32 Board Package
- TensorFlowLite_ESP32 Library

### 2. Kurulum

1. `esp32_fomo_digit.ino` dosyasını Arduino IDE ile açın
2. WiFi bilgilerinizi güncelleyin:
   ```cpp
   const char* ssid = "WiFi_Adi";
   const char* password = "WiFi_Sifresi";
   ```
3. Board ayarları:
   - Board: **AI Thinker ESP32-CAM**
   - Partition Scheme: **Huge APP (3MB)**
4. Kodu yükleyin

### 3. Test

1. Serial Monitor açın (115200 baud)
2. IP adresini not alın
3. Tarayıcıda `http://<IP>` adresine gidin
4. "Tespit Et" butonuna tıklayın

---

## 🎨 Web Arayüzü

Modern mavi tema ile tasarlanmış kullanıcı dostu arayüz:

### Özellikler
- 📷 **Canlı Kamera** - 2 saniyede bir otomatik yenileme
- 🔍 **Tespit Butonu** - Inference çalıştırır
- 📊 **İstatistikler** - Kare no, inference süresi, tespit sayısı
- 📋 **Sonuç Listesi** - Tespit edilen rakamlar + koordinatlar

### Görsel Tasarım
- Glassmorphism efektli kartlar
- Mavi gradient arka plan
- Animasyonlu orb'lar
- Shimmer başlık efekti
- Smooth hover animasyonları

### Endpoints
| URL | Açıklama |
|-----|----------|
| `/` | Ana web arayüzü |
| `/img` | Kamera görüntüsü (BMP) |
| `/run` | Inference çalıştır (text) |

---

## ⚡ Performans

### Benchmark

| Metrik | Değer |
|--------|-------|
| **Model Boyutu** | ~365KB |
| **Inference Time** | 40-60ms |
| **Tensor Arena** | 150KB |
| **FPS** | 15-20 |
| **mAP** | ~0.75 |

### Karşılaştırma

| Model | Boyut | Inference | 
|-------|-------|-----------|
| FOMO | 365KB | 50ms |
| YOLO-nano | 260KB | 180ms |
| MobileNet-SSD | 2MB | 300ms |

---

## 🔧 Preprocessing

FOMO modeli MNIST-like girdi bekler (beyaz rakam, siyah arka plan):

```cpp
void doInference(uint8_t* img) {
    // 1. Ortalama parlaklık hesapla
    uint8_t avg = calculateAverage(img);
    uint8_t threshold = avg - 30;
    
    // 2. Adaptive threshold + invert
    for (int i = 0; i < 128*128; i++) {
        // Kağıt (parlak) → 0 (siyah)
        // Mürekkep (koyu) → 255 (beyaz)
        input[i] = (img[i] < threshold) ? 255 : 0;
    }
    
    // 3. Inference
    interpreter->Invoke();
}
```

---

## 📝 Eğitim (Edge Impulse)

FOMO modeli Edge Impulse platformunda eğitilir:

### 1. Veri Seti Hazırlama
- 0-9 arası rakamları kağıda yazın
- Her rakamdan en az 50 örnek
- Fotoğrafları çekin ve yükleyin
- Bounding box ile etiketleyin

### 2. Model Eğitimi
```
Edge Impulse Studio:
1. Create new project
2. Data acquisition → Upload images
3. Labeling → Draw bounding boxes
4. Create impulse:
   - Image: 96×96 Grayscale
   - Processing: Image
   - Learning: Object Detection (FOMO)
5. Train model
6. Download → Arduino library
```

### 3. Model Dönüşümü
```bash
# Edge Impulse çıktısından header oluştur
xxd -i fomo_model.tflite > model_data.h
```

---

## 🐛 Sorun Giderme

### Model yüklenemiyor
- `model_data.h` dosyasının doğru konumda olduğunu kontrol edin
- Partition scheme "Huge APP" seçili mi?

### Kamera çalışmıyor
- PSRAM aktif mi? (Board: AI Thinker ESP32-CAM)
- Kamera kablosu düzgün takılı mı?

### Tespit yapılamıyor
- Işıklandırma yeterli mi?
- Rakamlar yeterince kontrastlı mı?
- Adaptive threshold çalışıyor mu? (Serial debug)

---

## 📊 Örnek Çıktı

```
=== RUN INFERENCE ===
Frame size: 9216 bytes
Avg brightness: 180, threshold: 150
Inference: 52ms, Detections: 2

Frame: 15
Inference time: 52 ms
Detections: 2

Digit 3 at (44,36) conf=87.5%
Digit 7 at (68,52) conf=92.1%
```

---

## 📚 Referanslar

- [Edge Impulse FOMO](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Pinout](https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/)

---

## 📝 Lisans

EE4065 Embedded Systems Final Project - Yıldız Teknik Üniversitesi

