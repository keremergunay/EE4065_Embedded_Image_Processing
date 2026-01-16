"""
EE4065 Homework 5 - TFLite Model Conversion
Convert Keras models to TFLite format and C++ header files for STM32

This script:
1. Loads the trained Keras models (kws_mlp.h5 and hdr_mlp.h5)
2. Converts them to TensorFlow Lite format
3. Generates C++ source/header files for embedding in STM32 firmware

The generated C++ files contain the model as a byte array that can be
loaded by TensorFlow Lite Micro on the STM32 Nucleo-F446RE.
"""

import os
import tensorflow as tf
from tensorflow import keras

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder

# Input model paths (from Models/ folder)
KWS_MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "kws_mlp.h5")
HDR_MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "hdr_mlp.h5")

# Output directory (in STM32/ folder)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "STM32", "tflite_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# TFLite to C++ Conversion Function
# ============================================================================

def convert_tflite2cc(tflite_model: bytes, output_path: str, model_name: str = "model", var_prefix: str = ""):
    """
    Convert TFLite model bytes to C++ source and header files.
    
    Parameters:
    -----------
    tflite_model : bytes
        The TFLite model as bytes
    output_path : str
        Output path without extension (will create .h and .cpp files)
    model_name : str
        Name to use for the model array variable
    var_prefix : str
        Prefix for variable names (e.g., "kws_" or "hdr_") to allow multiple models
    """
    hdr_filepath = output_path + ".h"
    src_filepath = output_path + ".cpp"
    hdr_filename = os.path.basename(hdr_filepath)
    
    arr_len = len(tflite_model)
    var_name = f"{var_prefix}converted_model_tflite" if var_prefix else "converted_model_tflite"
    var_len_name = f"{var_prefix}converted_model_tflite_len" if var_prefix else "converted_model_tflite_len"
    
    # Generate header file
    with open(hdr_filepath, "w") as hdr_file:
        hdr_file.write(f"// Auto-generated TFLite model header\n")
        hdr_file.write(f"// Model size: {arr_len} bytes\n\n")
        hdr_file.write(f"#ifndef {model_name.upper()}_H\n")
        hdr_file.write(f"#define {model_name.upper()}_H\n\n")
        hdr_file.write(f"extern const unsigned char {var_name}[];\n")
        hdr_file.write(f"extern const unsigned int {var_len_name};\n\n")
        hdr_file.write(f"#endif // {model_name.upper()}_H\n")
    
    # Generate source file
    with open(src_filepath, "w") as src_file:
        src_file.write(f"// Auto-generated TFLite model source\n")
        src_file.write(f"// Model size: {arr_len} bytes\n\n")
        src_file.write(f'#include "{hdr_filename}"\n\n')
        src_file.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        
        # Write bytes in groups of 12 per line for readability
        for i in range(0, arr_len, 12):
            chunk = tflite_model[i:i + 12]
            hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
            src_file.write(f"    {hex_values},\n")
        
        src_file.write("};\n\n")
        src_file.write(f"const unsigned int {var_len_name} = {arr_len};\n")
    
    print(f"  Generated: {hdr_filepath}")
    print(f"  Generated: {src_filepath}")
    print(f"  Model size: {arr_len} bytes ({arr_len/1024:.2f} KB)")


def convert_model(model_path: str, output_name: str, quantize: bool = False, var_prefix: str = ""):
    """
    Convert a Keras model to TFLite format and generate C++ files.
    
    Parameters:
    -----------
    model_path : str
        Path to the Keras .h5 model file
    output_name : str
        Base name for output files
    quantize : bool
        Whether to apply dynamic range quantization (reduces size)
    var_prefix : str
        Prefix for C variable names (e.g., "kws_" or "hdr_")
    """
    print(f"\nConverting: {model_path}")
    print("-" * 50)
    
    # Load Keras model
    model = keras.models.load_model(model_path, compile=False)
    print("  Model loaded successfully")
    model.summary()
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization if requested
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  Applying dynamic range quantization")
    
    # Convert to TFLite
    tflite_model = converter.convert()
    print(f"  Conversion successful")
    
    # Save .tflite file
    tflite_path = os.path.join(OUTPUT_DIR, f"{output_name}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved: {tflite_path}")
    
    # Generate C++ files
    cpp_path = os.path.join(OUTPUT_DIR, output_name)
    convert_tflite2cc(tflite_model, cpp_path, output_name, var_prefix)
    
    return tflite_model


# ============================================================================
# Main Conversion
# ============================================================================

print("=" * 60)
print("TFLite MODEL CONVERSION FOR STM32")
print("=" * 60)

# Convert Q1: Keyword Spotting Model
if os.path.exists(KWS_MODEL_PATH):
    print("\n" + "=" * 60)
    print("Q1: KEYWORD SPOTTING MODEL (kws_mlp)")
    print("=" * 60)
    kws_tflite = convert_model(KWS_MODEL_PATH, "kws_mlp", quantize=False, var_prefix="kws_")
    
    # Also create a quantized version (smaller, needed for F446RE)
    print("\n  Creating quantized version for STM32...")
    kws_tflite_quant = convert_model(KWS_MODEL_PATH, "kws_mlp_quant", quantize=True, var_prefix="kws_")
else:
    print(f"\nWARNING: KWS model not found at {KWS_MODEL_PATH}")
    print("  Run Q1_keyword_spotting.py first to train the model")

# Convert Q2: Handwritten Digit Recognition Model
if os.path.exists(HDR_MODEL_PATH):
    print("\n" + "=" * 60)
    print("Q2: HANDWRITTEN DIGIT RECOGNITION MODEL (hdr_mlp)")
    print("=" * 60)
    hdr_tflite = convert_model(HDR_MODEL_PATH, "hdr_mlp", quantize=False, var_prefix="hdr_")
    
    # Also create a quantized version (smaller, needed for F446RE)
    print("\n  Creating quantized version for STM32...")
    hdr_tflite_quant = convert_model(HDR_MODEL_PATH, "hdr_mlp_quant", quantize=True, var_prefix="hdr_")
else:
    print(f"\nWARNING: HDR model not found at {HDR_MODEL_PATH}")
    print("  Run Q2_handwritten_digit_recognition.py first to train the model")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")

if os.path.exists(OUTPUT_DIR):
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  {f} ({size/1024:.2f} KB)")

print("\n" + "-" * 60)
print("NOTE: TFLite files generated for reference.")
print("The simplified approach uses extract_weights.py instead.")
print("-" * 60)

