#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFLite to ESP32 C Header Converter
Question 4 - Multi-Model Digit Recognition

Converts TFLite models to C header files for ESP32-CAM
"""

import os
import tensorflow as tf
import numpy as np

# Input/Output directories
MODELS_DIR = 'd:/Projects/Embedded/Final_Project/Question4_MultiModel_Recognition/models'
OUTPUT_DIR = 'd:/Projects/Embedded/Final_Project/Question4_MultiModel_Recognition/esp32_cam/digit_recognition'

# Models to convert
MODELS = [
    'SqueezeNetMini',
    'MobileNetV2Mini', 
    'ResNet8',
    'EfficientNetMini'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_tflite_to_header(tflite_path, header_path, model_name):
    """Convert TFLite model to C header file"""
    
    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Get quantization parameters
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    input_shape = input_details['shape']
    input_dtype = input_details['dtype']
    input_quant = input_details.get('quantization', (0, 0))
    
    output_shape = output_details['shape']
    output_dtype = output_details['dtype']
    output_quant = output_details.get('quantization', (0, 0))
    
    # Handle quantization parameters
    if 'quantization_parameters' in input_details:
        input_scale = input_details['quantization_parameters'].get('scales', [1.0])[0]
        input_zp = input_details['quantization_parameters'].get('zero_points', [0])[0]
    else:
        input_scale = input_quant[0] if input_quant[0] != 0 else 1.0
        input_zp = input_quant[1]
    
    if 'quantization_parameters' in output_details:
        output_scale = output_details['quantization_parameters'].get('scales', [1.0])[0]
        output_zp = output_details['quantization_parameters'].get('zero_points', [0])[0]
    else:
        output_scale = output_quant[0] if output_quant[0] != 0 else 1.0
        output_zp = output_quant[1]
    
    # Generate variable name
    var_name = model_name.lower().replace('-', '_')
    
    # Write header file
    with open(header_path, 'w') as f:
        f.write(f'// {model_name} for ESP32-CAM - Handwritten Digit Recognition\n')
        f.write(f'// Model size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)\n')
        f.write(f'// Input shape: {list(input_shape)}, dtype: {input_dtype}\n')
        f.write(f'// Output shape: {list(output_shape)}, dtype: {output_dtype}\n')
        f.write(f'// Converted from: {os.path.basename(tflite_path)}\n\n')
        
        f.write(f'#ifndef {var_name.upper()}_MODEL_H\n')
        f.write(f'#define {var_name.upper()}_MODEL_H\n\n')
        f.write('#include <stdint.h>\n\n')
        
        # Model info
        f.write(f'#define {var_name.upper()}_INPUT_WIDTH {input_shape[1]}\n')
        f.write(f'#define {var_name.upper()}_INPUT_HEIGHT {input_shape[2]}\n')
        f.write(f'#define {var_name.upper()}_INPUT_CHANNELS {input_shape[3]}\n')
        f.write(f'#define {var_name.upper()}_NUM_CLASSES {output_shape[1]}\n\n')
        
        # Quantization parameters
        f.write(f'const float {var_name}_input_scale = {input_scale}f;\n')
        f.write(f'const int {var_name}_input_zero_point = {input_zp};\n')
        f.write(f'const float {var_name}_output_scale = {output_scale}f;\n')
        f.write(f'const int {var_name}_output_zero_point = {output_zp};\n\n')
        
        # Model data
        f.write(f'const unsigned int {var_name}_model_len = {len(model_data)};\n')
        f.write(f'alignas(8) const unsigned char {var_name}_model[] = {{\n')
        
        # Write bytes (16 per line)
        for i, byte in enumerate(model_data):
            if i % 16 == 0:
                f.write('  ')
            f.write(f'0x{byte:02x},')
            if i % 16 == 15:
                f.write('\n')
        
        if len(model_data) % 16 != 0:
            f.write('\n')
        
        f.write('};\n\n')
        f.write('#endif\n')
    
    print(f'✓ {model_name}: {len(model_data)/1024:.1f} KB')
    print(f'  Input: {list(input_shape)} ({input_dtype})')
    print(f'  Output: {list(output_shape)} ({output_dtype})')
    print(f'  Quantization: scale={input_scale:.6f}, zp={input_zp}')
    print()

def main():
    print('='*60)
    print('TFLite to ESP32 C Header Converter')
    print('='*60)
    print()
    
    for model_name in MODELS:
        tflite_path = os.path.join(MODELS_DIR, f'{model_name}.tflite')
        header_path = os.path.join(OUTPUT_DIR, f'{model_name.lower()}_model.h')
        
        if os.path.exists(tflite_path):
            convert_tflite_to_header(tflite_path, header_path, model_name)
        else:
            print(f'✗ {model_name}: TFLite file not found')
            print()
    
    print('='*60)
    print('Conversion complete!')
    print(f'Headers saved to: {OUTPUT_DIR}')
    print('='*60)

if __name__ == '__main__':
    main()
