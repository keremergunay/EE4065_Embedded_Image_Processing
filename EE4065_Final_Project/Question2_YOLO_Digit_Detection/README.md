# EE4065 Final Project - Question 2
# Handwritten Digit Detection with YOLO on ESP32-CAM

## 🎯 İki Model Eğitimi
Bu proje iki farklı YOLO modeli eğitir:

| Model | Veri Seti | Açıklama |
|-------|-----------|----------|
| **Model A** | Roboflow | Kendi el yazınızla eğitilmiş |
| **Model B** | MNIST → YOLO | Sentetik detection veri seti |

Her iki model de YOLOv5-nano kullanır ve ESP32-CAM'e sığar (~500KB INT8).

---

## 📁 Proje Yapısı

```
Question2_YOLO_Digit_Detection/
├── README.md                                    # Bu dosya
├── colab/
│   └── YOLO_Digit_Training.ipynb               # Google Colab notebook
├── dataset/
│   ├── DATASET_PREPARATION_GUIDE.md            # Veri seti hazırlama rehberi
│   ├── images/
│   │   ├── train/                              # Eğitim görselleri (%80)
│   │   └── val/                                # Doğrulama görselleri (%20)
│   └── labels/
│       ├── train/                              # Eğitim etiketleri (YOLO format)
│       └── val/                                # Doğrulama etiketleri
├── esp32_cam/
│   ├── ARDUINO_SETUP_GUIDE.md                  # Arduino IDE kurulum rehberi
│   ├── digit_detection/                        # ANA KOD (YOLO/TFLite)
│   │   ├── digit_detection.ino                 # Arduino ana kod
│   │   └── digit_model.h                       # TFLite model (placeholder)
│   └── digit_detection_simple/                 # ALTERNATİF (Basit CNN)
│       ├── digit_detection_simple.ino          # Basit CNN kodu
│       └── simple_digit_model.h                # CNN ağırlıkları (placeholder)
└── models/
    └── (eğitilmiş model dosyaları buraya gelecek)
```

## 🎯 İki Farklı Yaklaşım

### 1. YOLO ile Object Detection (Ana Yaklaşım)
- **Klasör:** `esp32_cam/digit_detection/`
- **Model:** YOLOv5-nano → TensorFlow Lite
- **Boyut:** ~500KB - 2MB
- **Özellik:** Görüntüde birden fazla rakam tespit edebilir
- **Dezavantaj:** Model boyutu büyük olabilir

### 2. Basit CNN ile Classification (Alternatif)
- **Klasör:** `esp32_cam/digit_detection_simple/`
- **Model:** Basit 4-layer CNN
- **Boyut:** ~50-100KB
- **Özellik:** Çok küçük, hızlı inference
- **Dezavantaj:** Sadece tek rakam sınıflandırma

## Adım Adım Rehber

### 1. Veri Seti Hazırlama

1. Kağıda 0-9 arası rakamları yazın (her rakamdan en az 50 adet)
2. Telefonla veya kamera ile fotoğraflarını çekin
3. Her rakamı ayrı ayrı kırpın veya aynı fotoğrafta etiketleyin
4. YOLO formatında etiketleme yapın (LabelImg veya Roboflow kullanabilirsiniz)

### 2. Model Eğitimi (Google Colab)

1. `colab/YOLO_Digit_Training.ipynb` dosyasını Google Colab'a yükleyin
2. GPU runtime'ı aktif edin (Runtime > Change runtime type > GPU)
3. Veri setinizi Colab'a yükleyin
4. Notebook'u çalıştırın

### 3. ESP32-CAM Kurulumu

1. Arduino IDE'yi açın
2. ESP32 board desteğini ekleyin
3. Gerekli kütüphaneleri yükleyin
4. `esp32_cam/digit_detection/digit_detection.ino` dosyasını açın
5. WiFi bilgilerinizi güncelleyin
6. Kodu ESP32-CAM'e yükleyin

## ESP32-CAM Sınırlamaları

- **Flash:** 4MB (model + kod için ~3MB kullanılabilir)
- **PSRAM:** 4MB (görüntü işleme için)
- **SRAM:** 520KB (çalışma belleği)

Bu sınırlamalar nedeniyle YOLOv5-nano (~1.9MB) veya benzeri küçük modeller kullanılmalıdır.

## 🔧 Gerekli Kütüphaneler

### Arduino IDE için:
- ESP32 Board Package (v2.0.0+)
- TensorFlow Lite Micro (Arduino_TensorFlowLite)
- ESP32 Camera Driver (ESP32 paketinde dahil)

### Python/Colab için:
- ultralytics (YOLOv5/v8)
- torch
- opencv-python
- tensorflow (model dönüşümü için)
- onnx, onnx-tf (dönüşüm için)

---

## 🚀 Hızlı Başlangıç

### Adım 1: Model Eğitimi (Google Colab)
1. `YOLO_Digit_Training.ipynb` dosyasını Colab'a yükleyin
2. **Runtime → GPU** seçin
3. Tüm hücreleri sırayla çalıştırın
4. İki model otomatik eğitilir:
   - **Model A:** Roboflow veri setiniz
   - **Model B:** MNIST'ten oluşturulan sentetik veri seti
5. `digit_model_roboflow.h` ve `digit_model_mnist.h` dosyalarını indirin

### Adım 2: ESP32-CAM'e Yükleme
1. İndirilen header dosyasını (`digit_model_roboflow.h` veya `digit_model_mnist.h`) `esp32_cam/digit_detection/` klasörüne kopyalayın
2. Dosya adını `digit_model.h` olarak değiştirin
3. `digit_detection.ino` dosyasını Arduino IDE ile açın
4. WiFi bilgilerinizi güncelleyin
5. ESP32-CAM'e yükleyin

### Adım 3: Test
1. Seri monitörden IP adresini alın
2. Tarayıcıda IP adresine gidin
3. Rakam tespitini test edin

---

## ⚠️ ESP32-CAM Sınırlamaları

| Kaynak | Değer | Not |
|--------|-------|-----|
| Flash Memory | 4MB | Model + kod için ~3MB |
| SRAM | 520KB | Runtime için |
| PSRAM | 4MB (varsa) | Görüntü buffer için |
| Max Model | ~2MB | INT8 quantization önerilir |

### Model Boyutu Önerileri:
- **YOLOv5-nano:** ~1.9MB (sınırda)
- **MobileNet-tiny:** ~500KB (uygun)
- **Basit CNN:** ~50-100KB (ideal)

---

## 📝 Dosya Açıklamaları

| Dosya | Açıklama |
|-------|----------|
| `YOLO_Digit_Training.ipynb` | Colab notebook - 2 model eğitimi |
| `digit_detection.ino` | ESP32-CAM ana kodu (TFLite) |
| `digit_model.h` | Eğitilmiş model verileri |
| `digit_detection_simple.ino` | Alternatif basit CNN kodu |
| `DATASET_PREPARATION_GUIDE.md` | Veri seti hazırlama rehberi |
| `ARDUINO_SETUP_GUIDE.md` | Arduino IDE kurulum rehberi |

---

## 🌐 Web Arayüzü

ESP32-CAM başarıyla çalıştığında web arayüzüne erişebilirsiniz:

- **Ana sayfa:** `http://<IP_ADRESI>/`
- **Stream:** `http://<IP_ADRESI>:81/stream`
- **Tespit API:** `http://<IP_ADRESI>/detect`

---

## 🔍 Sorun Giderme

### Model çok büyük
- INT8 quantization kullanın
- Daha küçük input boyutu deneyin (96x96 → 64x64)
- Alternatif basit CNN modelini kullanın

### Inference çok yavaş
- PSRAM'ın aktif olduğundan emin olun
- CPU frekansını 240MHz yapın
- Görüntü boyutunu küçültün

### WiFi bağlanmıyor
- 2.4GHz ağ kullandığınızdan emin olun
- SSID ve şifrede özel karakter kontrolü yapın

---

## 📚 Kaynaklar

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [ESP32-CAM Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/camera.html)
- [STM32 AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)