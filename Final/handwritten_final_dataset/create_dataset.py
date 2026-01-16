import numpy as np
import cv2
import os
import zipfile
import random
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='constant', cval=255).reshape(shape)

def generate_natural_digit(digit):
    img = np.ones((96, 96), dtype=np.uint8) * 255
    # El yazısı tipi fontlar seçelim
    font = random.choice([cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX])
    scale = random.uniform(1.8, 2.3)
    thick = random.randint(2, 4)
    (w, h), _ = cv2.getTextSize(str(digit), font, scale, thick)
    
    # Yazıyı yerleştir
    cv2.putText(img, str(digit), (48 - w//2 + random.randint(-8,8), 48 + h//2 + random.randint(-8,8)), 
                font, scale, 0, thick, cv2.LINE_AA)
    
    # El yazısı deformasyonu
    img = elastic_transform(img, alpha=random.uniform(35, 55), sigma=random.uniform(4, 5))
    
    # Rastgele döndürme
    M = cv2.getRotationMatrix2D((48, 48), random.uniform(-20, 20), 1)
    img = cv2.warpAffine(img, M, (96, 96), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    # Hafif bulanıklaştırma ve threshold ile kağıt üzerindeki mürekkep hissi
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    
    return img

# Klasör oluştur
output_dir = 'handwritten_final_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1000 adet görsel üret (0-9 arası her birinden 100 tane)
all_file_paths = []
for d in range(10):
    for i in range(100):
        img = generate_natural_digit(d)
        filename = f'digit_{d}_{i:03d}.png'
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        all_file_paths.append(filepath)

# ZIP dosyasına koy
zip_filename = 'dataset_el_yazisi_96x96.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in all_file_paths:
        zipf.write(file, os.path.basename(file))

print(f"Başarıyla {len(all_file_paths)} görsel oluşturuldu ve {zip_filename} dosyasına paketlendi.")