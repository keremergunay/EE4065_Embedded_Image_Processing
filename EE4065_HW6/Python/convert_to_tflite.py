"""
EE 4065 - Embedded Digital Image Processing - Homework 6
TFLite Conversion and C++ Export Script

This script converts trained Keras models to quantized TFLite format
and exports them as C++ header files for STM32 deployment.

Target: STM32 Nucleo-F446RE (128KB RAM, 512KB Flash)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
TFLITE_DIR = os.path.join(os.path.dirname(__file__), 'tflite_models')
CPP_EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'stm32_exports')

# Create directories
os.makedirs(TFLITE_DIR, exist_ok=True)
os.makedirs(CPP_EXPORT_DIR, exist_ok=True)

# Input shape for representative dataset
INPUT_SHAPE = (32, 32, 3)


def load_representative_data():
    """Load a subset of MNIST for quantization calibration"""
    (train_images, _), _ = keras.datasets.mnist.load_data()
    
    # Preprocess same as training
    train_images = np.expand_dims(train_images, axis=-1)
    train_images = np.repeat(train_images, 3, axis=-1)
    train_images = tf.image.resize(train_images, (32, 32)).numpy()
    train_images = train_images / 255.0
    
    return train_images.astype(np.float32)


def representative_dataset_gen(images, num_samples=1000, batch_size=1):
    """Generator for representative dataset"""
    for i in range(0, min(num_samples, len(images)), batch_size):
        yield [images[i:i+batch_size]]


def convert_to_tflite_int8(model_path, output_path, representative_data):
    """
    Convert Keras model to fully quantized INT8 TFLite model
    
    Args:
        model_path: Path to saved Keras model (.h5)
        output_path: Path for output TFLite model
        representative_data: Sample data for quantization calibration
    
    Returns:
        Size of TFLite model in bytes
    """
    print(f"\nConverting: {os.path.basename(model_path)}")
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Set target spec for INT8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Set representative dataset for full integer quantization
    def rep_dataset():
        for i in range(0, 1000, 32):
            yield [representative_data[i:i+32]]
    
    converter.representative_dataset = rep_dataset
    
    # Force INT8 input/output for full integer quantization
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    try:
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model)
        print(f"  TFLite model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")
        
        return tflite_model, model_size
    
    except Exception as e:
        print(f"  Error converting: {e}")
        return None, 0


def convert_tflite_to_cpp(tflite_model, output_path, model_name):
    """
    Convert TFLite model to C++ header file for STM32
    
    Args:
        tflite_model: TFLite model bytes
        output_path: Base path for output (without extension)
        model_name: Name for the model variable
    """
    header_path = output_path + ".h"
    source_path = output_path + ".cpp"
    header_filename = os.path.basename(header_path)
    
    # Clean model name for C variable
    var_name = model_name.replace('-', '_').replace('.', '_')
    
    model_size = len(tflite_model)
    
    # Write header file
    with open(header_path, 'w') as f:
        f.write(f"// Auto-generated TFLite model header\n")
        f.write(f"// Model: {model_name}\n")
        f.write(f"// Size: {model_size:,} bytes\n\n")
        f.write(f"#ifndef {var_name.upper()}_H\n")
        f.write(f"#define {var_name.upper()}_H\n\n")
        f.write(f"extern const unsigned char {var_name}_tflite[];\n")
        f.write(f"extern const unsigned int {var_name}_tflite_len;\n\n")
        f.write(f"#endif // {var_name.upper()}_H\n")
    
    # Write source file
    with open(source_path, 'w') as f:
        f.write(f'#include "{header_filename}"\n\n')
        f.write(f"// TFLite model data ({model_size:,} bytes)\n")
        f.write(f"alignas(8) const unsigned char {var_name}_tflite[] = {{\n")
        
        # Write data in rows of 12 bytes
        for i in range(0, model_size, 12):
            row = tflite_model[i:i+12]
            hex_values = ', '.join(f'0x{b:02x}' for b in row)
            f.write(f"    {hex_values},\n")
        
        f.write(f"}};\n\n")
        f.write(f"const unsigned int {var_name}_tflite_len = {model_size};\n")
    
    print(f"  C++ export: {header_path}, {source_path}")


def verify_tflite_model(tflite_path, sample_data):
    """Verify TFLite model works correctly"""
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"  Input: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        print(f"  Output: {output_details[0]['shape']}, {output_details[0]['dtype']}")
        
        # Test inference
        sample = sample_data[0:1]
        
        # Quantize input if needed
        if input_details[0]['dtype'] == np.uint8:
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            sample = (sample / input_scale + input_zero_point).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        if output_details[0]['dtype'] == np.uint8:
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        prediction = np.argmax(output[0])
        print(f"  Test prediction: {prediction}")
        
        return True
    
    except Exception as e:
        print(f"  Verification failed: {e}")
        return False


def estimate_arena_size(tflite_path):
    """Estimate tensor arena size needed for the model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get tensor details
        tensor_details = interpreter.get_tensor_details()
        
        # Calculate approximate arena size
        total_size = 0
        for tensor in tensor_details:
            tensor_size = np.prod(tensor['shape']) * np.dtype(tensor['dtype']).itemsize
            total_size += tensor_size
        
        # Add overhead (approximately 20%)
        arena_size = int(total_size * 1.2)
        
        print(f"  Estimated arena size: {arena_size:,} bytes ({arena_size/1024:.1f} KB)")
        
        return arena_size
    
    except Exception as e:
        print(f"  Could not estimate arena size: {e}")
        return 0


def main():
    """Main conversion function"""
    print("=" * 60)
    print("EE 4065 - Homework 6: TFLite Conversion")
    print("Converting trained models for STM32 Nucleo-F446RE")
    print("=" * 60)
    
    # Load representative data
    print("\nLoading representative data for quantization...")
    representative_data = load_representative_data()
    print(f"Loaded {len(representative_data)} samples")
    
    # Find all trained models
    if not os.path.exists(MODEL_SAVE_DIR):
        print(f"\nError: Model directory not found: {MODEL_SAVE_DIR}")
        print("Please run train_models.py first!")
        return
    
    model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith('.h5')]
    
    if not model_files:
        print(f"\nNo trained models found in {MODEL_SAVE_DIR}")
        print("Please run train_models.py first!")
        return
    
    print(f"\nFound {len(model_files)} trained models")
    
    # Convert each model
    results = {}
    
    for model_file in sorted(model_files):
        model_name = model_file.replace('.h5', '')
        model_path = os.path.join(MODEL_SAVE_DIR, model_file)
        tflite_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")
        cpp_path = os.path.join(CPP_EXPORT_DIR, model_name)
        
        # Convert to TFLite
        tflite_model, model_size = convert_to_tflite_int8(
            model_path, tflite_path, representative_data
        )
        
        if tflite_model is not None:
            # Verify model
            verify_tflite_model(tflite_path, representative_data)
            
            # Estimate arena size
            arena_size = estimate_arena_size(tflite_path)
            
            # Export to C++
            convert_tflite_to_cpp(tflite_model, cpp_path, model_name)
            
            results[model_name] = {
                'size': model_size,
                'arena': arena_size,
                'fits_f446re': model_size < 400000 and arena_size < 100000
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"\nSTM32 Nucleo-F446RE Constraints:")
    print(f"  Flash: 512 KB (model should be < ~400 KB)")
    print(f"  RAM: 128 KB (arena should be < ~100 KB)")
    print()
    print(f"{'Model':<25} {'TFLite Size':<15} {'Arena Size':<15} {'Fits F446RE':<12}")
    print("-" * 70)
    
    for name in sorted(results.keys()):
        r = results[name]
        fits = "✓ YES" if r['fits_f446re'] else "✗ NO"
        print(f"{name:<25} {r['size']/1024:>8.1f} KB     {r['arena']/1024:>8.1f} KB     {fits}")
    
    # Recommend best model
    fitting_models = [(n, r) for n, r in results.items() if r['fits_f446re']]
    if fitting_models:
        best = min(fitting_models, key=lambda x: x[1]['size'])
        print(f"\nRecommended for F446RE: {best[0]} ({best[1]['size']/1024:.1f} KB)")
    else:
        print("\nWarning: No models fit the F446RE constraints!")
        print("Consider using the 'Mini' variants which are designed for constrained MCUs.")
    
    print(f"\nTFLite models saved to: {TFLITE_DIR}")
    print(f"C++ exports saved to: {CPP_EXPORT_DIR}")
    print("\nNext step: Copy the C++ files to your STM32 project")


if __name__ == "__main__":
    main()

