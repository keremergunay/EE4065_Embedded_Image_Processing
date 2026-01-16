# Veri Seti Hazırlama Rehberi
## EE4065 Final Project - Question 2

Bu rehber, el yazısı rakam tespiti için veri seti hazırlamayı açıklar.

---

## 1. Veri Toplama

### Gerekli Malzemeler
- Beyaz kağıt (A4 veya daha küçük)
- Siyah kalem veya keçeli kalem
- Telefon veya kamera

### Yazım Kuralları
1. Her sayfaya 1-5 rakam yazın
2. Rakamları büyük ve net yazın (en az 3x3 cm)
3. Her rakamdan **en az 50 örnek** oluşturun
4. Farklı yazım stilleri kullanın:
   - Normal yazı
   - Eğik yazı
   - Kalın/ince kalem
   - Farklı boyutlar

### Fotoğraf Çekimi
- İyi aydınlatma sağlayın
- Kağıda dik açıyla çekin
- Gölge olmamasına dikkat edin
- 1080p veya üzeri çözünürlük kullanın

---

## 2. Etiketleme (Labeling)

### Araçlar
- **LabelImg** (Ücretsiz, masaüstü): https://github.com/tzutalin/labelImg
- **Roboflow** (Online, ücretsiz plan mevcut): https://roboflow.com
- **CVAT** (Online, açık kaynak): https://cvat.org

### LabelImg Kullanımı

1. LabelImg'ı indirin ve kurun:
```bash
pip install labelImg
labelImg
```

2. Görüntüleri açın (Open Dir)
3. YOLO formatını seçin (Change Save Format → YOLO)
4. Her rakamı dikdörtgen içine alın (Create RectBox veya 'w' tuşu)
5. Sınıf adını girin (0, 1, 2, ... 9)
6. Kaydedin (Save veya Ctrl+S)

### Roboflow Kullanımı (Önerilen)

1. https://roboflow.com adresine gidin
2. Ücretsiz hesap oluşturun
3. Yeni proje oluşturun (Object Detection)
4. Görüntüleri yükleyin
5. Online etiketleme aracını kullanın
6. YOLO formatında dışa aktarın

---

## 3. YOLO Format Yapısı

### Klasör Yapısı
```
dataset/
├── images/
│   ├── train/           # Eğitim görselleri (%80)
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/             # Doğrulama görselleri (%20)
│       ├── img101.jpg
│       ├── img102.jpg
│       └── ...
└── labels/
    ├── train/           # Eğitim etiketleri
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── val/             # Doğrulama etiketleri
        ├── img101.txt
        ├── img102.txt
        └── ...
```

### Etiket Dosyası Formatı
Her `.txt` dosyası karşılık gelen görüntüdeki nesneleri içerir:

```
<class_id> <x_center> <y_center> <width> <height>
```

- **class_id**: 0-9 arası rakam sınıfı
- **x_center**: Merkez x koordinatı (0-1 arası, normalize)
- **y_center**: Merkez y koordinatı (0-1 arası, normalize)
- **width**: Kutu genişliği (0-1 arası, normalize)
- **height**: Kutu yüksekliği (0-1 arası, normalize)

### Örnek Etiket Dosyası
`img001.txt`:
```
5 0.45 0.32 0.12 0.18
3 0.72 0.68 0.10 0.15
7 0.25 0.51 0.11 0.16
```

Bu dosya şu anlama gelir:
- Rakam "5": merkez (0.45, 0.32), boyut (0.12, 0.18)
- Rakam "3": merkez (0.72, 0.68), boyut (0.10, 0.15)
- Rakam "7": merkez (0.25, 0.51), boyut (0.11, 0.16)

---

## 4. Veri Seti Bölme

Toplam veri setini şu oranlarda bölün:
- **Eğitim (train)**: %80
- **Doğrulama (val)**: %20

### Python ile Bölme
```python
import os
import shutil
import random

def split_dataset(source_images, source_labels, dest_dir, train_ratio=0.8):
    """Veri setini train ve val olarak böler"""
    
    # Klasörleri oluştur
    for split in ['train', 'val']:
        os.makedirs(f'{dest_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{dest_dir}/labels/{split}', exist_ok=True)
    
    # Dosya listesi
    images = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    # Bölme noktası
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Dosyaları kopyala
    for img in train_images:
        shutil.copy(f'{source_images}/{img}', f'{dest_dir}/images/train/{img}')
        label = img.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(f'{source_labels}/{label}'):
            shutil.copy(f'{source_labels}/{label}', f'{dest_dir}/labels/train/{label}')
    
    for img in val_images:
        shutil.copy(f'{source_images}/{img}', f'{dest_dir}/images/val/{img}')
        label = img.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(f'{source_labels}/{label}'):
            shutil.copy(f'{source_labels}/{label}', f'{dest_dir}/labels/val/{label}')
    
    print(f'Train: {len(train_images)} görüntü')
    print(f'Val: {len(val_images)} görüntü')

# Kullanım
split_dataset('raw_images', 'raw_labels', 'dataset')
```

---

## 5. Veri Artırma (Data Augmentation)

Veri setinizi büyütmek için augmentation kullanın:

```python
import cv2
import numpy as np
import os

def augment_image(img_path, label_path, output_dir, num_augments=5):
    """Görüntüyü çeşitli yöntemlerle artırır"""
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Etiketleri oku
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    for i in range(num_augments):
        aug_img = img.copy()
        aug_labels = labels.copy()
        
        # Rastgele dönüşümler
        if np.random.random() > 0.5:
            # Yatay çevirme
            aug_img = cv2.flip(aug_img, 1)
            aug_labels = [flip_label_horizontal(l) for l in aug_labels]
        
        if np.random.random() > 0.5:
            # Parlaklık değişimi
            factor = np.random.uniform(0.7, 1.3)
            aug_img = np.clip(aug_img * factor, 0, 255).astype(np.uint8)
        
        if np.random.random() > 0.5:
            # Gaussian blur
            ksize = np.random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)
        
        if np.random.random() > 0.5:
            # Gürültü ekleme
            noise = np.random.normal(0, 10, aug_img.shape).astype(np.int16)
            aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Kaydet
        out_name = f'{base_name}_aug{i}'
        cv2.imwrite(f'{output_dir}/images/{out_name}.jpg', aug_img)
        with open(f'{output_dir}/labels/{out_name}.txt', 'w') as f:
            f.writelines(aug_labels)

def flip_label_horizontal(label_line):
    """Yatay çevirme için etiketi günceller"""
    parts = label_line.strip().split()
    class_id = parts[0]
    x_center = 1.0 - float(parts[1])  # x'i çevir
    y_center = parts[2]
    width = parts[3]
    height = parts[4]
    return f'{class_id} {x_center:.6f} {y_center} {width} {height}\n'
```

---

## 6. Veri Seti Kontrolü

Etiketlerin doğru olduğunu kontrol edin:

```python
import cv2
import os

def visualize_labels(img_path, label_path):
    """Etiketleri görüntü üzerinde gösterir"""
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(img, str(class_id), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[class_id], 2)
    
    cv2.imshow('Labels', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Kullanım
visualize_labels('dataset/images/train/img001.jpg', 
                 'dataset/labels/train/img001.txt')
```

---

## 7. Zip Dosyası Oluşturma

Veri setini Colab'a yüklemek için zip'leyin:

### Windows (PowerShell)
```powershell
Compress-Archive -Path dataset -DestinationPath digit_dataset.zip
```

### Linux/Mac
```bash
zip -r digit_dataset.zip dataset/
```

---

## 8. Minimum Gereksinimler

| Özellik | Minimum | Önerilen |
|---------|---------|----------|
| Toplam görüntü | 500 | 1000+ |
| Her sınıf için | 50 | 100+ |
| Görüntü boyutu | 640x480 | 1280x720 |
| Format | JPG/PNG | JPG |

---

## 9. İpuçları

1. **Çeşitlilik önemli**: Farklı el yazıları, farklı kalem kalınlıkları
2. **Arka plan çeşitliliği**: Farklı kağıt tipleri, çizgili/kareli
3. **Aydınlatma çeşitliliği**: Farklı ışık koşulları
4. **Negatif örnekler**: Rakam olmayan görüntüler de ekleyin
5. **Denge**: Her sınıftan eşit sayıda örnek

---

## Sonraki Adım

Veri setinizi hazırladıktan sonra:
1. `digit_dataset.zip` dosyasını oluşturun
2. Google Colab'a yükleyin
3. `YOLO_Digit_Training.py` scriptini çalıştırın
4. Eğitilmiş modeli indirin
5. ESP32-CAM'e yükleyin
