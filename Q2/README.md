# EE4065 Final Project - Question 2
# 🔢 YOLO Tabanlı El Yazısı Rakam Tespiti (ESP32-CAM)

Bu proje, ESP32-CAM üzerinde çalışan **YOLOv5-nano** tabanlı gerçek zamanlı el yazısı rakam tespit sistemidir. Model, kamera görüntüsünde birden fazla rakamı eş zamanlı olarak tespit edebilir ve konumlarını belirleyebilir.

---

## 🎯 Proje Özeti

| Özellik | Değer |
|---------|-------|
| **Platform** | ESP32-CAM (AI-Thinker) |
| **Model** | YOLOv5-nano (TensorFlow Lite INT8) |
| **Giriş Boyutu** | 96×96 Grayscale |
| **Çıkış** | 6×6 Grid × 15 değer (bbox + 10 sınıf) |
| **Model Boyutu** | ~260KB (INT8 Quantized) |
| **Inference Süresi** | ~150-200ms |
| **Desteklenen Sınıflar** | 0-9 arası rakamlar |

---

## 📁 Proje Yapısı

```
Q2/
├── README.md                    # Bu dosya
├── MNIST RESULT.jpg             # MNIST modeli test sonucu
├── ROBOFLOW RESLUT.jpg          # Roboflow modeli test sonucu
├── esp32_cam/
│   └── digit_detection/
│       ├── digit_detection.ino  # 🎯 Ana ESP32-CAM kodu
│       ├── yolo_model_mnist.h   # MNIST tabanlı model
│       └── yolo_model_roboflow.h # Roboflow tabanlı model
├── kaggle/
│   ├── train_yolo_nano.py       # Model eğitim scripti
│   ├── convert_q2_headers.py    # TFLite → Header dönüştürücü
│   └── ...                      # Diğer eğitim dosyaları
└── dataset/
    └── ...                      # Eğitim veri seti
```

---

## 🚀 Hızlı Başlangıç

### 1. Arduino IDE Kurulumu

```
1. Arduino IDE'yi açın
2. Preferences → Additional Board URLs:
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
3. Tools → Board → Boards Manager → "ESP32" ara ve yükle
4. Library Manager → "TensorFlowLite_ESP32" yükle
```

### 2. Kodu Yükleme

1. `esp32_cam/digit_detection/digit_detection.ino` dosyasını açın
2. WiFi bilgilerinizi güncelleyin:
   ```cpp
   const char* sta_ssid = "WiFi_Adi";
   const char* sta_password = "WiFi_Sifresi";
   ```
3. Model seçimi yapın:
   ```cpp
   #define USE_ROBOFLOW_MODEL false  // true = Roboflow, false = MNIST
   ```
4. Board ayarları:
   - Board: **AI Thinker ESP32-CAM**
   - Upload Speed: **115200**
   - Partition Scheme: **Huge APP (3MB)**
5. Kodu yükleyin (GPIO0 → GND bağlayıp reset)

### 3. Test Etme

1. Serial Monitor'den IP adresini alın (115200 baud)
2. Tarayıcıda `http://<IP_ADRESI>` adresine gidin
3. "Fotoğraf Çek ve Tespit Et" butonuna tıklayın

---

## 🎨 Web Arayüzü Özellikleri

### Ana Sayfa (`/`)
- **Canlı Kamera Görüntüsü** - Tespit sonrası bounding box'lar ile
- **Fotoğraf Çek Butonu** - Görüntü yakalar ve inference çalıştırır
- **Flash Kontrolü** - LED flaşı açıp kapatır
- **Tespit Sonuçları** - Bulunan rakamlar, güven skoru, koordinatlar

### Threshold Ayarları (Yeni!)
- **Aktif/Pasif** - Binary thresholding'i açıp kapatır
- **Değer Slider** - 0-255 arası threshold değeri
- **Invert Toggle** - Beyaz kağıt/siyah yazı için tersine çevirme
- **Debug Görüntü** - Model'in gördüğü 96x96 preprocessed görüntü

### API Endpoints
| Endpoint | Açıklama |
|----------|----------|
| `/` | Web arayüzü |
| `/capture` | Fotoğraf çek + tespit yap (JSON) |
| `/snapshot` | Son çekilen fotoğraf (JPEG) |
| `/detect` | Sadece tespit yap (JSON) |
| `/stream` | MJPEG canlı yayın (port 81) |
| `/flash?state=1` | Flash kontrolü |
| `/threshold?val=128&en=1&inv=0` | Threshold ayarları |
| `/debug_input` | Debug görüntü (BMP) |

---

## 🧠 Model Detayları

### İki Model Seçeneği

| Model | Kaynak | Özellik |
|-------|--------|---------|
| **MNIST** | Sintetik MNIST veri seti | Temiz rakamlar için optimize |
| **Roboflow** | Kendi el yazınız | Gerçek dünya koşulları için |

### YOLO-Nano Çıktı Formatı

```
Output Shape: [1, 6, 6, 2, 15]
- 6×6 Grid (görüntü 16 piksellik hücrelere bölünür)
- 2 Anchor per cell
- 15 değer per anchor:
  - [0]: Objectness score
  - [1-4]: x, y, w, h (normalize)
  - [5-14]: Class probabilities (0-9 rakamları)
```

### Preprocessing Pipeline

```
1. Kameradan GRAYSCALE görüntü al (QQVGA: 160×120)
2. Bilinear interpolation ile 96×96'ya resize et
3. Contrast Stretch: min-max normalization
4. (Opsiyonel) Binary Thresholding: değer ≥ T → 255, değil → 0
5. (Opsiyonel) Invert: Beyaz kağıt → siyah, siyah yazı → beyaz
6. INT8 Quantization: scale=0.00378, zero_point=-128
```

---

## ⚙️ Teknik Özellikler

### Donanım Gereksinimleri

| Bileşen | Değer |
|---------|-------|
| **Board** | ESP32-CAM (AI-Thinker) |
| **Flash** | 4MB |
| **PSRAM** | 4MB (gerekli) |
| **CPU** | 240MHz Dual-Core |
| **Kamera** | OV2640 / GC2145 |

### Bellek Kullanımı

| Kaynak | Kullanım | Kapasite |
|--------|----------|----------|
| Flash (Kod+Model) | ~1.5MB | 4MB |
| PSRAM (Tensor Arena) | 200KB | 4MB |
| SRAM | ~50KB | 520KB |

---

## 🔧 Sorun Giderme

### Kamera başlatılamadı
- PSRAM'ın aktif olduğunu kontrol edin (Board: AI Thinker ESP32-CAM)
- Kamera kablosunu kontrol edin
- Power supply yeterli mi? (5V, en az 500mA)

### Model inference başarısız
- Partition scheme: "Huge APP (3MB)" seçili mi?
- Model header dosyası doğru konumda mı?

### Bounding box'lar yanlış yerde
- Threshold ayarlarını kontrol edin
- Debug görüntüyü inceleyin (`/debug_input`)
- Invert toggle'ı deneyin

### WiFi bağlanamıyor
- 2.4GHz ağ kullanın (5GHz desteklenmiyor)
- AP Mode'a geçmeyi deneyin (`USE_AP_MODE true`)

---

## 📊 Test Sonuçları

### MNIST Model
- **mAP@0.5**: 0.85+
- **Inference Time**: ~180ms
- **En İyi Performans**: Temiz, kontrastlı rakamlar

### Roboflow Model
- **mAP@0.5**: 0.75+
- **Inference Time**: ~180ms
- **En İyi Performans**: Gerçek el yazısı

---

## 📚 Referanslar

- [YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Documentation](https://docs.espressif.com/projects/esp-idf/)
- [Edge Impulse FOMO](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices)

---

## 📝 Lisans

EE4065 Embedded Systems Final Project - Yıldız Teknik Üniversitesi
