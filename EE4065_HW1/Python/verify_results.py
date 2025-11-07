import cv2
import numpy as np

# --- AYARLAR ---
# C kodunuzdaki header'ı oluştururken kullandığınız ayarların AYNISI olmalı
INPUT_IMAGE_FILE = "D:\Projects\Embedded\PC_Python\mandrill.tiff"  # DEĞİŞTİRİN: Kullandığınız orijinal resim
IMG_WIDTH = 160
IMG_HEIGHT = 120
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT
# --- AYARLAR BİTTİ ---

print(f"--- '{INPUT_IMAGE_FILE}' için Beklenen Sonuçlar Hesaplaniyor ---")
print(f"Cozunurluk: {IMG_WIDTH}x{IMG_HEIGHT}\n")

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
    print("Orijinal Görüntü (GRAYSCALE_IMG_ARRAY):")
    print([hex(p) for p in original_image_flat[:20]])
    print("-" * 20)

    # --- 2a. Negatif Görüntü ---
    img_negative = 255 - original_image_flat
    print("Negatif Görüntü (g_negativeImage):")
    print([hex(p) for p in img_negative[:20]])
    print("-" * 20)

    # --- 2b. Eşikleme ---
    threshold_value = 128
    # C'deki (p > 128) ? 255 : 0 mantığının aynısı
    img_threshold = np.where(original_image_flat > threshold_value, 255, 0).astype(np.uint8)
    print(f"Eşikle T={threshold_value} (g_thresholdImage):")
    print([hex(p) for p in img_threshold[:20]])
    print("-" * 20)
    
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

    print("Gamma (gamma=3.0) (g_gammaImage_3):")
    print([hex(p) for p in img_gamma_3[:20]])
    print("-" * 20)
    
    print("Gamma (gamma=1/3) (g_gammaImage_1_3):")
    print([hex(p) for p in img_gamma_1_3[:20]])
    print("-" * 20)

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
    print(f"Parçalı Doğrusal (r1={r1}, r2={r2}) (g_piecewiseImage):")
    print([hex(p) for p in img_piecewise[:20]])
    print("-" * 20)

except Exception as e:
    print(f"Bir hata olustu: {e}")
    print("Lütfen betik içindeki INPUT_IMAGE_FILE ve boyut ayarlarını kontrol edin.")
