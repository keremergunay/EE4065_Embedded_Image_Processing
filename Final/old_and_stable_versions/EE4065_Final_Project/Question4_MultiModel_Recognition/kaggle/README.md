# ESP32-CAM Digit Recognition - Model Training

Bu klasör, ESP32-CAM için 4 mini CNN modelinin eğitim scriptlerini içerir.

## Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `train_digit_models.py` | Ana eğitim scripti - 4 model eğitir |
| `convert_to_headers.py` | TFLite → ESP32 C header dönüştürücü |

## Modeller

| Model | Boyut | Özellik |
|-------|-------|---------|
| SqueezeNetMini | ~55 KB | Fire modülleri |
| MobileNetV2Mini | ~110 KB | Depthwise separable conv |
| ResNet8 | ~100 KB | Residual connections |
| EfficientNetMini | ~110 KB | MBConv + SE |

## Kullanım

### 1. Modelleri Eğit (Kaggle/Colab'da çalıştır)

```bash
pip install tensorflow numpy matplotlib
python train_digit_models.py
```

### 2. ESP32 Header'larına Dönüştür

```bash
python convert_to_headers.py
```

### 3. Header'ları ESP32'ye Kopyala

Oluşturulan header dosyaları otomatik olarak `../esp32_cam/digit_recognition/` klasörüne kaydedilir.

## Preprocessing Uyumu

ESP32 preprocessing çıktısı:
- **Beyaz rakam (255)** siyah arka plan (0) üzerinde
- **32x32x3** boyutunda (RGB, grayscale tekrarlanmış)
- **uint8** veri tipi

Eğitim verileri aynı formatta hazırlanır.

## Kaggle Notebook

Kaggle'da çalıştırmak için:

```python
!pip install tensorflow
!python train_digit_models.py
```

Eğitim sonrası `trained_models/` klasöründen TFLite dosyalarını indirin.
