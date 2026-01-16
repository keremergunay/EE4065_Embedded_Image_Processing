#!/usr/bin/env python3
"""
TFLite to ESP32 C Header Converter (Q2)
Converts both YOLO-Nano models to separate C headers.

Usage:
  python convert_q2_headers.py
  
Outputs:
  - yolo_model_mnist.h (synthetic MNIST model)
  - yolo_model_roboflow.h (Roboflow real data model)
"""

import os
import tensorflow as tf

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'esp32_cam', 'digit_detection')

# Model configurations
MODELS = {
    'mnist': {
        'tflite': 'yolo_nano_synthetic_int8',
        'header': 'yolo_model_mnist',
        'prefix': 'yolo_mnist'
    },
    'roboflow': {
        'tflite': 'yolo_nano_roboflow_int8',
        'header': 'yolo_model_roboflow',
        'prefix': 'yolo_roboflow'
    }
}

def convert_single_model(config):
    tflite_path = os.path.join(MODELS_DIR, f"{config['tflite']}.tflite")
    header_path = os.path.join(OUTPUT_DIR, f"{config['header']}.h")
    prefix = config['prefix']
    
    if not os.path.exists(tflite_path):
        print(f"⚠ Model not found: {tflite_path}")
        return False

    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Get quantization parameters
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_details['shape']
    output_shape = output_details['shape']
    
    def get_quant_params(details):
        if 'quantization_parameters' in details:
            scales = details['quantization_parameters'].get('scales', [1.0])
            zps = details['quantization_parameters'].get('zero_points', [0])
            if len(scales) > 0: return scales[0], zps[0]
        return 1.0, 0

    in_scale, in_zp = get_quant_params(input_details)
    out_scale, out_zp = get_quant_params(output_details)
    
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate header guard from prefix
    guard = f"{prefix.upper()}_MODEL_H"
    
    # Write header
    with open(header_path, 'w') as f:
        f.write(f'// YOLO-Nano Model: {config["tflite"]}\n')
        f.write(f'// Size: {len(model_data)/1024:.1f} KB\n')
        f.write(f'// Input: {list(input_shape)}\n')
        f.write(f'// Output: {list(output_shape)}\n\n')
        
        f.write(f'#ifndef {guard}\n')
        f.write(f'#define {guard}\n\n')
        
        f.write('// Quantization Parameters\n')
        f.write(f'const float {prefix}_input_scale = {in_scale}f;\n')
        f.write(f'const int {prefix}_input_zero_point = {in_zp};\n')
        f.write(f'const float {prefix}_output_scale = {out_scale}f;\n')
        f.write(f'const int {prefix}_output_zero_point = {out_zp};\n\n')
        
        # Model Data
        f.write(f'const unsigned int {prefix}_model_len = {len(model_data)};\n')
        f.write(f'alignas(8) const unsigned char {prefix}_model[] = {{\n')
        
        for i, byte in enumerate(model_data):
            if i % 16 == 0: f.write('  ')
            f.write(f'0x{byte:02x},')
            if i % 16 == 15: f.write('\n')
            
        if len(model_data) % 16 != 0: f.write('\n')
        
        f.write('};\n\n')
        f.write('#endif\n')
        
    print(f"✓ {config['header']}.h ({len(model_data)/1024:.1f} KB)")
    print(f"    Input: scale={in_scale:.6f}, zp={in_zp}")
    print(f"    Output: scale={out_scale}, zp={out_zp}")
    return True

def main():
    print("=" * 50)
    print("  Converting TFLite Models to ESP32 Headers")
    print("=" * 50)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    converted = 0
    for name, config in MODELS.items():
        print(f"\n[{name.upper()}]")
        if convert_single_model(config):
            converted += 1
    
    print(f"\n{'=' * 50}")
    print(f"  Converted {converted}/{len(MODELS)} models")
    print("=" * 50)
    
    if converted > 0:
        print("\nTo use in ESP32 code:")
        print('  #include "yolo_model_mnist.h"    // MNIST-based model')
        print('  #include "yolo_model_roboflow.h" // Roboflow model')

if __name__ == '__main__':
    main()
