# Arduino IDE Kurulum Rehberi
## ESP32-CAM Digit Detection

---

## 1. Arduino IDE Kurulumu

### Arduino IDE 2.x İndirme
1. https://www.arduino.cc/en/software adresine gidin
2. Arduino IDE 2.x'i indirin ve kurun

---

## 2. ESP32 Board Paketi Kurulumu

### Adım 1: Board Manager URL Ekleme
1. Arduino IDE'yi açın
2. **File → Preferences** (veya Ctrl+Comma)
3. "Additional Board Manager URLs" alanına ekleyin:
```
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
4. OK'a tıklayın

### Adım 2: ESP32 Paketini Kurma
1. **Tools → Board → Boards Manager**
2. "esp32" aratın
3. "esp32 by Espressif Systems" paketini bulun
4. **Install** butonuna tıklayın (v2.0.0 veya üzeri)
5. Kurulum tamamlanana kadar bekleyin

### Adım 3: Board Seçimi
1. **Tools → Board → ESP32 Arduino**
2. **AI Thinker ESP32-CAM** seçin

---

## 3. TensorFlow Lite Micro Kurulumu

### Seçenek A: Arduino Library Manager (Önerilen)
1. **Sketch → Include Library → Manage Libraries**
2. "TensorFlow Lite" aratın
3. "Arduino_TensorFlowLite" kütüphanesini kurun

### Seçenek B: Manuel Kurulum
1. https://github.com/tensorflow/tflite-micro-arduino-examples adresinden indirin
2. ZIP dosyasını Arduino libraries klasörüne çıkartın:
   - Windows: `C:\Users\<username>\Documents\Arduino\libraries\`
   - Mac: `~/Documents/Arduino/libraries/`
   - Linux: `~/Arduino/libraries/`

### Seçenek C: ESP32 için Optimize Edilmiş TFLite
ESP-NN ile hızlandırılmış versiyon:
```bash
# Libraries klasörüne git
cd ~/Documents/Arduino/libraries

# Clone et
git clone https://github.com/eloquentarduino/EloquentTinyML.git
```

---

## 4. Gerekli Kütüphaneler

### Otomatik Kurulum (Library Manager)
1. **Sketch → Include Library → Manage Libraries**
2. Aşağıdakileri kurun:
   - `ArduinoJson` - JSON işleme
   - `ESP Async WebServer` - Web sunucu (opsiyonel)

### ESP32 Kamera Kütüphanesi
ESP32 paketinde dahili olarak gelir, ekstra kurulum gerekmez.

---

## 5. Proje Dosyalarını Açma

1. `digit_detection.ino` dosyasını Arduino IDE ile açın
2. Aynı klasörde `digit_model.h` dosyasının olduğundan emin olun
3. WiFi bilgilerini güncelleyin:
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

---

## 6. Derleme Ayarları

### Tools Menüsü Ayarları
| Ayar | Değer |
|------|-------|
| Board | AI Thinker ESP32-CAM |
| CPU Frequency | 240MHz (WiFi/BT) |
| Flash Frequency | 80MHz |
| Flash Mode | QIO |
| Flash Size | 4MB (32Mb) |
| Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |
| Core Debug Level | None |
| PSRAM | Enabled |

**ÖNEMLİ:** Partition Scheme'i "Huge APP" olarak ayarlayın! 
Model dosyası büyük olduğu için standart partition yetmeyebilir.

---

## 7. ESP32-CAM'e Yükleme

### Bağlantı Şeması (FTDI Programmer)
```
ESP32-CAM          FTDI
---------          ----
5V         -->     VCC
GND        -->     GND
U0R (GPIO3) -->    TX
U0T (GPIO1) -->    RX
GPIO0      -->     GND (sadece upload sırasında)
```

### Upload Adımları
1. GPIO0'ı GND'ye bağlayın (programming mode)
2. ESP32-CAM'i RESET'leyin (RST pinine kısa GND)
3. Arduino IDE'de doğru COM portunu seçin
4. **Upload** butonuna tıklayın (veya Ctrl+U)
5. "Connecting..." mesajı göründüğünde RST'ye basın
6. Upload tamamlanınca GPIO0 bağlantısını çıkarın
7. ESP32-CAM'i RESET'leyin

### Alternatif: ESP32-CAM-MB Kartı
Eğer ESP32-CAM-MB (programlayıcı) kartınız varsa:
1. ESP32-CAM'i MB kartına takın
2. USB ile bilgisayara bağlayın
3. Doğrudan upload edin (GPIO0 bağlantısı otomatik)

---

## 8. Seri Monitör

Upload tamamlandıktan sonra:
1. **Tools → Serial Monitor** (veya Ctrl+Shift+M)
2. Baud rate: **115200**
3. Çıktıyı kontrol edin:
```
===================================
ESP32-CAM Digit Detection
EE4065 Final Project - Question 2
===================================

PSRAM bulundu!
Kamera başarıyla başlatıldı!
TensorFlow Lite başlatılıyor...
Input  -> Tip: 1, Boyut: [1, 96, 96, 3]
Output -> Tip: 1, Boyut: [1, 10]
TensorFlow Lite başarıyla başlatıldı!
WiFi'ye bağlanılıyor....
WiFi Bağlandı!
IP Adresi: 192.168.1.xxx
Web Arayüzü: http://192.168.1.xxx
```

---

## 9. Sorun Giderme

### "Brownout detector was triggered" Hatası
- Güç kaynağı yetersiz
- 5V/2A adaptör kullanın
- USB hub kullanmayın, doğrudan bağlayın

### "Camera probe failed" Hatası
- Kamera modülü düzgün takılmamış
- Kamera kablosunu kontrol edin
- Kamerayı yeniden takın

### "Failed to allocate memory" Hatası
- Model çok büyük
- PSRAM'ı aktif edin: Tools → PSRAM → Enabled
- Partition scheme'i "Huge APP" yapın

### Upload Başarısız
- GPIO0 GND'ye bağlı mı?
- Doğru COM portu seçili mi?
- FTDI driver yüklü mü?
- RST'ye bastınız mı?

### WiFi Bağlanmıyor
- SSID ve şifre doğru mu?
- 2.4GHz ağ mı? (5GHz desteklenmiyor)
- Router'a yakın mısınız?

---

## 10. Test Etme

### Seri Port Üzerinden
Seri monitörde komutlar:
- `d` - Rakam tespiti yap
- `f` - Flash LED aç/kapat
- `i` - Sistem bilgisi
- `h` - Yardım

### Web Arayüzü
1. Seri monitörden IP adresini alın
2. Tarayıcıda IP adresini açın
3. Canlı görüntü ve tespit sonuçlarını görün

---

## 11. Bellek Kullanımı

Derleme sonrası beklenen değerler:
```
Sketch uses XXXXX bytes (XX%) of program storage space.
Global variables use XXXXX bytes (XX%) of dynamic memory.
```

| Bileşen | Yaklaşık Boyut |
|---------|----------------|
| Kamera driver | ~100 KB |
| WiFi stack | ~200 KB |
| HTTP server | ~50 KB |
| TFLite runtime | ~150 KB |
| Model (INT8) | 100KB - 2MB |
| Tensor arena | 100 KB |

**Toplam:** ~700KB + Model boyutu

---

## Sonraki Adımlar

1. ✅ Arduino IDE kurulumu
2. ✅ ESP32 paketi kurulumu
3. ✅ Kütüphaneler kurulumu
4. ⏳ Model eğitimi (Google Colab)
5. ⏳ digit_model.h güncelleme
6. ⏳ ESP32-CAM'e yükleme
7. ⏳ Test etme
