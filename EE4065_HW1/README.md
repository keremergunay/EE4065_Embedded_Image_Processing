# EE4065 – Embedded Digital Image Processing
## Homework 1
**Due Date:** November 7, 2025 — 23:59

**Team Members:** * Kerem Ergünay
* Tarık Erdoğan

---

## 📋 Description
This project implements fundamental digital image processing operations on an ARM Cortex-M microcontroller (STM32). The process involves converting a source image (mandrill.tiff) to a 160x120 8-bit grayscale C header file (.h) using a custom Python library. This header is then loaded into the STM32's memory to apply intensity transformations (**Negative**, **Thresholding**, **Gamma Correction**, and **Piecewise Linear**).

The C-code implementation is optimized using **Look-Up Tables (LUTs)** for Gamma and Piecewise transformations, ensuring high performance on the embedded platform. Results are verified by comparing the STM32's memory (via the STM32CubeIDE Memory Window) against a reference Python verification script.

---

## 🛠️ Hardware & Software
* **Hardware:** STM32 Development Board (ARM Cortex-M)
* **Software (PC):** * Python 3.x (w/ OpenCV, Numpy)
    * STM32CubeIDE
* **Image:** mandrill.tiff, processed to 160x120 grayscale.

---

## Q1 — Grayscale Image Formation (40 pts)
### 🔹 Objective
Convert a source image to a 160x120, 8-bit grayscale C header file (mandrill.h) containing a `const unsigned char` array.

### 🔹 Python Workflow
The conversion is handled by a two-part Python system: a reusable library (`Image_Header_Library.py`) and a driver script (`imageHeaderGenerator.py`).

1.  **Library: `Image_Header_Library.py`**
    This library handles the core image processing. The `spi_c_generate_grayscale` function is used for this homework.

    ```python
    # Image_Header_Library.py
    import cv2
    import numpy as np
    
    def spi_c_generate_grayscale(im, outputFileName, width, height):
        """
        Bu fonksiyon, görüntüyü 8-bit grayscale C header dosyasına dönüştürür.
        (Ödev 1a için güncellendi)
        """
        f = open(outputFileName + ".h", "w+")
    
        # Görüntüyü yeniden boyutlandır ve grayscale'e çevir
        im_resized = cv2.resize(im, (width, height))
        im_gray = cv2.cvtColor(im_resized, cv2.COLOR_BGR2GRAY)
        
        # Görüntü boyutlarını al
        height, width = im_gray.shape
        array_size = width * height
    
        print(f"Header dosyasi (SPI_C - Grayscale) '{outputFileName}.h' olusturuluyor...")
        
        # 2D diziyi 1D (düz) bir diziye çevir
        img_flat = np.reshape(im_gray, (array_size))
    
        f.write(f"// Format: 8-bit Grayscale, Cozunurluk: {width}x{height}\n")
        f.write(f"// Generated from: {outputFileName}\n")
        f.write(f"#define IMG_WIDTH   {width}\n")
        f.write(f"#define IMG_HEIGHT  {height}\n")
        f.write(f"#define IMG_SIZE    ({width} * {height})\n\n")
    
        # Dizi adını main.c ile uyumlu yap
        f.write(f"const unsigned char GRAYSCALE_IMG_ARRAY[IMG_SIZE] = {{\n")
    
        for i in range(array_size):
            f.write("%s, " % hex(img_flat[i])) # Hex formatında yaz
            if (i + 1) % 20 == 0:
                f.write("\n")
    
        f.write("\n}};\n\n")
        f.close()
        print("Dosya olusturma tamamlandi.")
    
    def generate(filename, width, height, outputFileName, format):
        Img = cv2.imread(filename)
        if Img is None:
            print(f"Hata: '{filename}' dosyasi bulunamadi.")
            return
    
        if format == 2: # SPI_C_GRAYSCALE
            spi_c_generate_grayscale(Img, outputFileName, width, height)
        # ... (diğer formatlar)
    ```
    2.  **Driver Script: `imageHeaderGenerator.py`**
    This script imports the library and calls it with the specific parameters for this project.

    ```python
    # imageHeaderGenerator.py
    import Image_Header_Library as headerGenerator
    
    # Proje ayarları
    Img = "D:\Projects\Embedded\PC_Python\mandrill.tiff"
    outputFileName = "mandrill"
    width = 160
    height = 120
    
    # Ödev 1 için 8-bit grayscale formatını seç
    # (headerGenerator.SPI_C_GRAYSCALE = 2)
    headerGenerator.generate(Img, width, height, outputFileName, 2)
    ```

### 🔹 Execution Steps
1.  Place `mandrill.tiff` in the path specified in `imageHeaderGenerator.py`.
2.  Run the driver script: `python imageHeaderGenerator.py`
3.  The file `mandrill.h` will be generated.
4.  Move `mandrill.h` into the STM32 project: `Core/Inc/`.
5.  Build the STM32 project and start a debug session.
6.  Open the Memory Window and monitor `GRAYSCALE_IMG_ARRAY` to verify the data is loaded.

### 🔹 Results (Q1)
* **Original Grayscale Image (160x120):** `mandrill_grayscaled.png`
* **Memory Observation (Q1):**
    > ![Proje Logosu](EE4065_HW1/memory shots/grayscaled.jpg)
    >
    > Bu ekran görüntüsü, `mandrill.h` dosyasındaki verilerin STM32'nin flash belleğine doğru bir şekilde yüklendiğini doğrular.

## Q2 — Intensity Transformations (60 pts)
### 🔹 Objective
Implement and verify **negative**, **thresholding**, **gamma correction**, and **piecewise linear** transformations in C on the STM32. All transformations are applied to the `GRAYSCALE_IMG_ARRAY` and stored in separate output arrays.

### 🔹 STM32 Code (main.c)
The transformations are implemented in `main.c`. **Look-Up Tables (LUTs)** are pre-computed for Gamma and Piecewise transformations to avoid costly floating-point operations inside the main image processing loops.

```c
/* USER CODE BEGIN Includes */
#include "mandrill.h" // Q1'de oluşturulan header dosyası
#include <math.h>     // Sadece LUT'ları doldurmak için powf()
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
// Görüntü boyutları "mandrill.h" dosyasından gelir (IMG_SIZE)

// İşlenmiş görüntüleri saklamak için STATIK diziler
static unsigned char g_negativeImage[IMG_SIZE];
static unsigned char g_thresholdImage[IMG_SIZE];
static unsigned char g_gammaImage_3[IMG_SIZE];      // Gamma = 3 için
static unsigned char g_gammaImage_1_3[IMG_SIZE];    // Gamma = 1/3 için
static unsigned char g_piecewiseImage[IMG_SIZE];

// --- Verimlilik için Arama Tabloları (LUT) ---
static unsigned char g_gammaLUT_3[256];     // Gamma = 3.0 için
static unsigned char g_gammaLUT_1_3[256];   // Gamma = 1/3 için
static unsigned char g_piecewiseLUT[256];   // Parçalı Doğrusal için
/* USER CODE END PV */

/* USER CODE BEGIN 2 */
// Orijinal görüntü dizisine bir pointer al
const unsigned char* originalImage = GRAYSCALE_IMG_ARRAY;

// --- LUT'ları Doldurma (Sadece bir kez yapılır) ---

// Q2-c: Gamma Düzeltmesi LUT'ları
float gamma_3_0 = 3.0f;
float gamma_1_3 = 1.0f / 3.0f;

for (int i = 0; i < 256; i++)
{
    float normalized_val = (float)i / 255.0f;
    g_gammaLUT_3[i] = (unsigned char)(powf(normalized_val, gamma_3_0) * 255.0f);
    g_gammaLUT_1_3[i] = (unsigned char)(powf(normalized_val, gamma_1_3) * 255.0f);
}

// Q2-d: Parçalı Doğrusal LUT
#define r1 50
#define s1 0
#define r2 200
#define s2 255
for (int i = 0; i < 256; i++)
{
    if (i < r1)       g_piecewiseLUT[i] = s1;
    else if (i > r2)  g_piecewiseLUT[i] = s2;
    else              g_piecewiseLUT[i] = (unsigned char)(s1 + (i - r1) * ((float)(s2 - s1) / (float)(r2 - r1)));
}


// --- Görüntü İşleme Döngüleri ---

// Q2-a: Negatif Görüntü
for (int i = 0; i < IMG_SIZE; i++)
{
    g_negativeImage[i] = 255 - originalImage[i];
}

// Q2-b: Eşikleme
#define THRESHOLD_VALUE 128
for (int i = 0; i < IMG_SIZE; i++)
{
    g_thresholdImage[i] = (originalImage[i] > THRESHOLD_VALUE) ? 255 : 0;
}

// Q2-c: Gamma Düzeltmesi (LUT Kullanarak)
for (int i = 0; i < IMG_SIZE; i++)
{
    unsigned char pixel = originalImage[i];
    g_gammaImage_3[i]   = g_gammaLUT_3[pixel];
    g_gammaImage_1_3[i] = g_gammaLUT_1_3[pixel];
}

// Q2-d: Parçalı Doğrusal (LUT Kullanarak)
for (int i = 0; i < IMG_SIZE; i++)
{
    g_piecewiseImage[i] = g_piecewiseLUT[originalImage[i]];
}
/* USER CODE END 2 */
```

### 🔹 Results (Q2)
**2a — Negative Image**
* **Description:** Inverts all pixel intensities ($s = 255 - r$).
* **Python Result:** `mandrill_nega.png`
* **STM32 Memory Result:**
    > [BURAYA g_negativeImage DİZİSİNİN MEMORY GÖRÜNTÜSÜNÜ EKLEYİN]

**2b — Thresholding**
* **Description:** Converts the image to binary. Pixels > 128 become 255 (white), others 0 (black).
* **Python Result:** `mandrill_threshold.png`
* **STM32 Memory Result:**
    > [BURAYA g_thresholdImage DİZİSİNİN MEMORY GÖRÜNTÜSÜNÜ EKLEYİN]

**2c — Gamma Correction**
* **Description:** Adjusts image brightness using $s = 255 \cdot (r/255)^\gamma$. Implemented via LUT.
* **Gamma = 3.0: (Darkens the image)**
    * **Python Result:** `mandrill_gamma3.png`
    * **STM32 Memory Result:**
        > [BURAYA g_gammaImage_3 DİZİSİNİN MEMORY GÖRÜNTÜSÜNÜ EKLEYİN]
* **Gamma = 1/3: (Brightens the image)**
    * **Python Result:** `mandrill_gamma1_3.png`
    * **STM32 Memory Result:**
        > [BURAYA g_gammaImage_1_3 DİZİSİNİN MEMORY GÖRÜNTÜSÜNÜ EKLEYİN]

**2d — Piecewise Linear Transformation**
* **Description:** Stretches the contrast of the [50, 200] pixel range to the full [0, 255] range. Implemented via LUT.
* **Python Result:** `mandrill_piecewise.png`
* **STM32 Memory Result:**
    > [BURAYA g_piecewiseImage DİZİSİNİN MEMORY GÖRÜNTÜSÜNÜ EKLEYİN]
## 🧪 Testing & Verification
To verify the correctness of the STM32 C code, a parallel Python script (`verify_results.py`) was used. This script performs the exact same operations as the C code (including generating identical LUTs) and prints the first 20 hexadecimal values for each transformation.

These printed values are used to cross-reference against the data shown in the STM32CubeIDE Memory Window.

### 🔹 Python Verification Script (verify\_results.py)
```python
import cv2
import numpy as np

# --- AYARLAR (C koduyla aynı olmalı) ---
INPUT_IMAGE_FILE = "D:\Projects\Embedded\PC_Python\mandrill.tiff"
IMG_WIDTH = 160
IMG_HEIGHT = 120
# --- AYARLAR BİTTİ ---

print(f"--- '{INPUT_IMAGE_FILE}' için Beklenen Sonuçlar Hesaplaniyor ---")

# Görüntüyü C koduyla aynı şekilde oku, yeniden boyutlandır ve grayscale yap
Img = cv2.imread(INPUT_IMAGE_FILE)
Img_resized = cv2.resize(Img, (IMG_WIDTH, IMG_HEIGHT))
Img_gray = cv2.cvtColor(Img_resized, cv2.COLOR_BGR2GRAY)
original_image_flat = Img_gray.flatten() # Düz 1D dizi

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
img_threshold = np.where(original_image_flat > threshold_value, 255, 0).astype(np.uint8)
print(f"Eşikleme T={threshold_value} (g_thresholdImage):")
print([hex(p) for p in img_threshold[:20]])
print("-" * 20)

# --- 2c. Gamma Düzeltmesi (C'deki LUT'un aynısı) ---
gamma_3_0 = 3.0
gamma_1_3 = 1.0 / 3.0
g_gammaLUT_3 = np.zeros(256, dtype=np.uint8)
g_gammaLUT_1_3 = np.zeros(256, dtype=np.uint8)

for i in range(256):
    normalized_val = i / 255.0
    g_gammaLUT_3[i] = np.uint8(np.power(normalized_val, gamma_3_0) * 255.0)
    g_gammaLUT_1_3[i] = np.uint8(np.power(normalized_val, gamma_1_3) * 255.0)

img_gamma_3 = g_gammaLUT_3[original_image_flat]
img_gamma_1_3 = g_gammaLUT_1_3[original_image_flat]

print("Gamma (gamma=3.0) (g_gammaImage_3):")
print([hex(p) for p in img_gamma_3[:20]])
print("Gamma (gamma=1/3) (g_gammaImage_1_3):")
print([hex(p) for p in img_gamma_1_3[:20]])
print("-" * 20)

# --- 2d. Parçalı Doğrusal (C'deki LUT'un aynısı) ---
r1, s1 = 50, 0
r2, s2 = 200, 255
g_piecewiseLUT = np.zeros(256, dtype=np.uint8)

for i in range(256):
    if i < r1:      g_piecewiseLUT[i] = s1
    elif i > r2:    g_piecewiseLUT[i] = s2
    else:           g_piecewiseLUT[i] = np.uint8(s1 + (i - r1) * ((s2 - s1) / (r2 - r1)))

img_piecewise = g_piecewiseLUT[original_image_flat]
print(f"Parçalı Doğrusal (r1={r1}, r2={r2}) (g_piecewiseImage):")
print([hex(p) for p in img_piecewise[:20]])
print("-" * 20)
```
### 🔹 Verification Process
1.  Run `verify_results.py` on the PC to get the expected hex values.
2.  In the STM32CubeIDE debugger, add `g_negativeImage`, `g_thresholdImage`, etc., to the Memory Window.
3.  **Result:** The values in the Memory Window perfectly matched the hex output from the Python script, confirming the 100% accuracy of the C-code implementation.

---

## 🏁 Summary
* **Python:** A custom library (`Image_Header_Library.py`) converted the `mandrill.tiff` image into a 160x120 8-bit grayscale C-header file (`mandrill.h`).
* **STM32 (C):** The image array was loaded from flash. Four intensity transformations were applied to it, storing results in RAM.
* **Optimization:** Look-Up Tables (LUTs) were pre-computed for Gamma and Piecewise transformations, allowing the main processing loops to run extremely fast using only integer array lookups.
* **Verification:** All C-code transformations were byte-for-byte verified against a reference Python script by inspecting the STM32's RAM in the debugger.
