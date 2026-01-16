import cv2
import numpy as np
from PIL import Image
# --- AYARLAR ---
# C kodunuzdaki header'ı oluştururken kullandığınız ayarların AYNISI olmalı
INPUT_IMAGE_FILE = "C:\mandrill.tiff"  # DEĞİŞTİRİN: Kullandığınız orijinal resim
IMG_WIDTH = 160
IMG_HEIGHT = 120
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT
# --- AYARLAR BİTTİ ---


# Görüntüyü C koduyla aynı şekilde oku, yeniden boyutlandır ve grayscale yap
try:
    Img = cv2.imread(INPUT_IMAGE_FILE)
    if Img is None:
        raise FileNotFoundError(f"'{INPUT_IMAGE_FILE}' bulunamadi.")
        
    Img_resized = cv2.resize(Img, (IMG_WIDTH, IMG_HEIGHT))
    Img_gray = cv2.cvtColor(Img_resized, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü düz 1D dizi yap (C'deki gibi)
    original_image_flat = Img_gray.flatten()

    # --- 1. Orijinal Görüntü ---

    img_bytes = bytearray(original_image_flat)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_ori.png")


    # --- 2a. Negatif Görüntü ---
    img_negative = 255 - original_image_flat

    
    img_bytes = bytearray(img_negative)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_nega.png")


    # --- 2b. Eşikleme ---
    threshold_value = 128
    # C'deki (p > 128) ? 255 : 0 mantığının aynısı
    img_threshold = np.where(original_image_flat > threshold_value, 255, 0).astype(np.uint8)
    img_bytes = bytearray(img_threshold)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_threshold.png")

    
    # --- 2c. Gamma Düzeltmesi (LUT Yöntemiyle) ---
    gamma_3_0 = 3.0
    gamma_1_3 = 1.0 / 3.0
    
    # C kodundaki LUT'un aynısını oluştur
    g_gammaLUT_3 = np.zeros(256, dtype=np.uint8)
    g_gammaLUT_1_3 = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        normalized_val = i / 255.0
        
        corrected_val_3_0 = np.power(normalized_val, gamma_3_0)
        g_gammaLUT_3[i] = np.uint8(corrected_val_3_0 * 255.0)
        
        corrected_val_1_3 = np.power(normalized_val, gamma_1_3)
        g_gammaLUT_1_3[i] = np.uint8(corrected_val_1_3 * 255.0)

    # Görüntüleri LUT kullanarak işle
    img_gamma_3 = g_gammaLUT_3[original_image_flat]
    img_gamma_1_3 = g_gammaLUT_1_3[original_image_flat]

    
    img_bytes = bytearray(img_gamma_3)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_gamma3.png")

   
    img_bytes = bytearray(img_gamma_1_3)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_gamma1_3.png")
    # --- 2d. Parçalı Doğrusal ---
    # C kodundaki ayarların aynısı (r1=50, s1=0, r2=200, s2=255)
    r1, s1 = 50, 0
    r2, s2 = 200, 255
    g_piecewiseLUT = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        if i < r1:
            g_piecewiseLUT[i] = s1
        elif i > r2:
            g_piecewiseLUT[i] = s2
        else:
            g_piecewiseLUT[i] = np.uint8(s1 + (i - r1) * ((s2 - s1) / (r2 - r1)))

    img_piecewise = g_piecewiseLUT[original_image_flat]
   
    img_bytes = bytearray(img_piecewise)
    img = Image.frombytes('L', (160, 120), img_bytes)
    img.save("mandrill_piecewise.png")

except Exception as e:
    print(f"Bir hata olustu: {e}")
    print("Lütfen betik içindeki INPUT_IMAGE_FILE ve boyut ayarlarını kontrol edin.")
