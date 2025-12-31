"""
EE4065 Homework 5 - Weight Extraction
Extract weights from trained Keras models and export as C header files for STM32

This script loads the trained MLP models and exports their weights/biases
as C arrays that can be directly used in embedded inference code.
"""

import os
import numpy as np
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder

def export_weights_to_c(model_path: str, output_path: str, prefix: str):
    """
    Extract weights from Keras model and export to C header file.
    
    Parameters:
    -----------
    model_path : str
        Path to the Keras .h5 model file
    output_path : str
        Path for output .h file
    prefix : str
        Prefix for variable names (e.g., 'kws_' or 'hdr_')
    """
    print(f"\nProcessing: {model_path}")
    print("-" * 50)
    
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    model.summary()
    
    # Open output file
    with open(output_path, 'w') as f:
        # Header guard
        guard_name = os.path.basename(output_path).replace('.', '_').upper()
        f.write(f"// Auto-generated from {os.path.basename(model_path)}\n")
        f.write(f"// MLP Weights for STM32 Inference\n\n")
        f.write(f"#ifndef {guard_name}\n")
        f.write(f"#define {guard_name}\n\n")
        
        # Extract layer info
        layer_count = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) == 0:
                continue  # Skip layers without weights (e.g., Input, Flatten)
            
            w = weights[0]  # Weights matrix
            b = weights[1]  # Bias vector
            
            print(f"  Layer {layer_count}: {layer.name}")
            print(f"    Weights: {w.shape}, Bias: {b.shape}")
            
            # Write dimensions
            f.write(f"// Layer {layer_count}: {layer.name}\n")
            f.write(f"// Input: {w.shape[0]}, Output: {w.shape[1]}\n")
            f.write(f"#define {prefix.upper()}LAYER{layer_count}_INPUT_SIZE {w.shape[0]}\n")
            f.write(f"#define {prefix.upper()}LAYER{layer_count}_OUTPUT_SIZE {w.shape[1]}\n\n")
            
            # Write weights (flattened, row-major)
            f.write(f"static const float {prefix}layer{layer_count}_weights[{w.size}] = {{\n")
            w_flat = w.flatten()
            for i in range(0, len(w_flat), 8):
                chunk = w_flat[i:i+8]
                f.write("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",\n")
            f.write("};\n\n")
            
            # Write biases
            f.write(f"static const float {prefix}layer{layer_count}_bias[{b.size}] = {{\n")
            f.write("    " + ", ".join(f"{v:.8f}f" for v in b) + "\n")
            f.write("};\n\n")
            
            layer_count += 1
        
        # Write number of layers
        f.write(f"#define {prefix.upper()}NUM_LAYERS {layer_count}\n\n")
        
        f.write(f"#endif // {guard_name}\n")
    
    print(f"  Generated: {output_path}")
    print(f"  Total layers with weights: {layer_count}")


# ============================================================================
# Main
# ============================================================================

print("=" * 60)
print("WEIGHT EXTRACTION FOR STM32 CUSTOM INFERENCE")
print("=" * 60)

# Directories
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
STM32_DIR = os.path.join(PROJECT_DIR, "STM32")
os.makedirs(STM32_DIR, exist_ok=True)

# Q1: Keyword Spotting Model
kws_model_path = os.path.join(MODELS_DIR, "kws_mlp.h5")
kws_output_path = os.path.join(STM32_DIR, "kws_weights.h")

if os.path.exists(kws_model_path):
    export_weights_to_c(kws_model_path, kws_output_path, "kws_")
else:
    print(f"\nWARNING: KWS model not found at {kws_model_path}")
    print("  Run Q1_keyword_spotting.py first")

# Q2: Handwritten Digit Recognition Model
hdr_model_path = os.path.join(MODELS_DIR, "hdr_mlp.h5")
hdr_output_path = os.path.join(STM32_DIR, "hdr_weights.h")

if os.path.exists(hdr_model_path):
    export_weights_to_c(hdr_model_path, hdr_output_path, "hdr_")
else:
    print(f"\nWARNING: HDR model not found at {hdr_model_path}")
    print("  Run Q2_handwritten_digit_recognition.py first")

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print("""
Generated files in STM32/ folder:
  - kws_weights.h
  - hdr_weights.h

Copy these to your STM32 project's Core/Inc/ folder.
""")
