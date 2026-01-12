// ===================================================
// Simple CNN Model Weights for Digit Recognition
// ===================================================
// 
// Bu dosya basit bir CNN modelin ağırlıklarını içerir.
// Model MNIST veri seti üzerinde eğitilmiştir.
// 
// Mimari:
// - Input: 28x28x1 (grayscale)
// - Conv1: 3x3, 8 filters -> 26x26x8
// - MaxPool: 2x2 -> 13x13x8
// - Conv2: 3x3, 16 filters -> 11x11x16
// - MaxPool: 2x2 -> 5x5x16
// - FC1: 400 -> 128
// - FC2: 128 -> 10
// 
// Toplam parametre: ~60K
// Tahmini model boyutu: ~240KB (float32)
// INT8 ile: ~60KB
// 
// ===================================================

#ifndef SIMPLE_DIGIT_MODEL_H
#define SIMPLE_DIGIT_MODEL_H

#include <Arduino.h>

// ===================================================
// Conv1: 3x3x1x8 kernel + 8 bias
// Boyut: 3*3*1*8 + 8 = 80 parametre
// ===================================================
const float conv1_weights[] PROGMEM = {
    // Filter 0 (3x3x1)
    0.1f, 0.2f, 0.1f,
    0.0f, 0.3f, 0.0f,
    -0.1f, 0.1f, -0.1f,
    // Filter 1
    0.2f, 0.1f, 0.2f,
    0.1f, 0.4f, 0.1f,
    0.0f, -0.1f, 0.0f,
    // Filter 2
    -0.1f, 0.3f, -0.1f,
    0.2f, 0.2f, 0.2f,
    0.1f, 0.1f, 0.1f,
    // Filter 3
    0.1f, -0.1f, 0.1f,
    0.2f, 0.5f, 0.2f,
    0.1f, 0.0f, 0.1f,
    // Filter 4
    0.0f, 0.2f, 0.0f,
    0.1f, 0.3f, 0.1f,
    0.1f, 0.2f, 0.1f,
    // Filter 5
    0.1f, 0.1f, 0.1f,
    0.0f, 0.4f, 0.0f,
    -0.1f, 0.2f, -0.1f,
    // Filter 6
    0.2f, 0.0f, 0.2f,
    0.1f, 0.3f, 0.1f,
    0.0f, 0.1f, 0.0f,
    // Filter 7
    -0.1f, 0.2f, -0.1f,
    0.1f, 0.5f, 0.1f,
    0.1f, 0.1f, 0.1f
};

const float conv1_bias[] PROGMEM = {
    0.01f, 0.01f, 0.01f, 0.01f,
    0.01f, 0.01f, 0.01f, 0.01f
};

// ===================================================
// Conv2: 3x3x8x16 kernel + 16 bias
// Boyut: 3*3*8*16 + 16 = 1168 parametre
// ===================================================
// NOT: Bu placeholder değerlerdir. 
// Gerçek model eğitimi sonrası güncellenmelidir.
const float conv2_weights[3 * 3 * 8 * 16] PROGMEM = {
    // 1152 değer - placeholder olarak rastgele başlatılmış
    // Gerçek modelden sonra güncellenecek
    0.05f, 0.10f, 0.05f, 0.00f, 0.15f, 0.00f, -0.05f, 0.05f, -0.05f,
    0.10f, 0.05f, 0.10f, 0.05f, 0.20f, 0.05f, 0.00f, -0.05f, 0.00f,
    // ... (kalan değerler eğitim sonrası gelecek)
    // Şimdilik sıfırlarla doldurulmuş
};

const float conv2_bias[] PROGMEM = {
    0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f,
    0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f
};

// ===================================================
// FC1: 400 -> 128
// Boyut: 400*128 + 128 = 51328 parametre
// ===================================================
// NOT: Bu çok büyük bir dizi. Gerçek implementasyonda
// ayrı bir dosyada veya SPIFFS'te saklanmalı.
const float fc1_weights[400 * 128] PROGMEM = {
    // 51200 değer - placeholder
    // Eğitim sonrası güncellenecek
    0.0f  // Placeholder - gerçek değerler model eğitiminden gelecek
};

const float fc1_bias[128] PROGMEM = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

// ===================================================
// FC2: 128 -> 10
// Boyut: 128*10 + 10 = 1290 parametre
// ===================================================
const float fc2_weights[128 * 10] PROGMEM = {
    // 1280 değer - placeholder
    // Eğitim sonrası güncellenecek
    0.0f  // Placeholder
};

const float fc2_bias[10] PROGMEM = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

// ===================================================
// Model Bilgileri
// ===================================================
#define MODEL_INPUT_SIZE 28
#define MODEL_NUM_CLASSES 10
#define MODEL_TOTAL_PARAMS 53866

/*
 * ===================================================
 * KULLANIM:
 * ===================================================
 * 
 * 1. Google Colab'da basit CNN modelini eğitin
 * 2. Model ağırlıklarını dışa aktarın
 * 3. Bu dosyadaki placeholder değerleri gerçek 
 *    değerlerle değiştirin
 * 
 * Colab'da ağırlıkları dışa aktarma:
 * 
 * ```python
 * import numpy as np
 * 
 * def export_weights_to_c(model, output_file):
 *     with open(output_file, 'w') as f:
 *         for layer in model.layers:
 *             weights = layer.get_weights()
 *             if len(weights) > 0:
 *                 w = weights[0].flatten()
 *                 f.write(f"// {layer.name} weights\n")
 *                 f.write("const float {}_weights[] PROGMEM = {{\n".format(layer.name))
 *                 for i, val in enumerate(w):
 *                     f.write(f"    {val:.6f}f,\n")
 *                 f.write("};\n\n")
 *                 
 *                 if len(weights) > 1:
 *                     b = weights[1].flatten()
 *                     f.write("const float {}_bias[] PROGMEM = {{\n".format(layer.name))
 *                     for val in b:
 *                         f.write(f"    {val:.6f}f,\n")
 *                     f.write("};\n\n")
 * 
 * export_weights_to_c(model, 'simple_digit_model.h')
 * ```
 * 
 * ===================================================
 */

#endif // SIMPLE_DIGIT_MODEL_H
