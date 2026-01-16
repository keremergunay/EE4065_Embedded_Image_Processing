"""
EE 4065 - Embedded Digital Image Processing - Homework 6
TFLite Model Evaluation Script

This script evaluates the quantized TFLite models to verify
accuracy before deploying to STM32.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

TFLITE_DIR = os.path.join(os.path.dirname(__file__), 'tflite_models')


def load_test_data():
    """Load and preprocess MNIST test data"""
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # Preprocess same as training
    test_images = np.expand_dims(test_images, axis=-1)
    test_images = np.repeat(test_images, 3, axis=-1)
    test_images = tf.image.resize(test_images, (32, 32)).numpy()
    test_images = test_images / 255.0
    
    return test_images.astype(np.float32), test_labels


def evaluate_tflite_model(tflite_path, test_images, test_labels, num_samples=1000):
    """Evaluate a TFLite model on test data"""
    model_name = os.path.basename(tflite_path).replace('.tflite', '')
    print(f"\nEvaluating: {model_name}")
    
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Get quantization parameters
    input_scale = input_details['quantization'][0]
    input_zero_point = input_details['quantization'][1]
    output_scale = output_details['quantization'][0]
    output_zero_point = output_details['quantization'][1]
    
    correct = 0
    total_time = 0
    
    # Limit samples for faster evaluation
    num_samples = min(num_samples, len(test_images))
    
    for i in range(num_samples):
        image = test_images[i:i+1]
        label = test_labels[i]
        
        # Quantize input if INT8
        if input_details['dtype'] == np.uint8:
            image = (image / input_scale + input_zero_point).astype(np.uint8)
        
        # Run inference
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        elapsed = time.perf_counter() - start
        total_time += elapsed
        
        # Dequantize output if INT8
        if output_details['dtype'] == np.uint8:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        prediction = np.argmax(output[0])
        if prediction == label:
            correct += 1
        
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{num_samples}")
    
    accuracy = correct / num_samples * 100
    avg_time_ms = total_time / num_samples * 1000
    
    # Get model size
    model_size = os.path.getsize(tflite_path)
    
    return {
        'name': model_name,
        'accuracy': accuracy,
        'avg_time_ms': avg_time_ms,
        'model_size_kb': model_size / 1024
    }


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("EE 4065 - Homework 6: TFLite Model Evaluation")
    print("=" * 60)
    
    # Load test data
    print("\nLoading MNIST test data...")
    test_images, test_labels = load_test_data()
    print(f"Loaded {len(test_images)} test samples")
    
    # Find TFLite models
    if not os.path.exists(TFLITE_DIR):
        print(f"\nNo TFLite models found in {TFLITE_DIR}")
        print("Please run convert_to_tflite.py first!")
        return
    
    tflite_files = [f for f in os.listdir(TFLITE_DIR) if f.endswith('.tflite')]
    
    if not tflite_files:
        print(f"\nNo TFLite models found in {TFLITE_DIR}")
        return
    
    print(f"\nFound {len(tflite_files)} TFLite models")
    
    # Evaluate each model
    results = []
    for tflite_file in sorted(tflite_files):
        tflite_path = os.path.join(TFLITE_DIR, tflite_file)
        try:
            result = evaluate_tflite_model(tflite_path, test_images, test_labels)
            results.append(result)
        except Exception as e:
            print(f"  Error evaluating {tflite_file}: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Inference':<15} {'Size':<12}")
    print(f"{'':25} {'':12} {'(PC, ms)':<15} {'(KB)':<12}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['name']:<25} {r['accuracy']:>6.2f}%      {r['avg_time_ms']:>8.3f}       {r['model_size_kb']:>8.1f}")
    
    # Note about STM32 inference time
    print("\n" + "-" * 70)
    print("Note: Actual inference time on STM32 Nucleo-F446RE will be different.")
    print("The Cortex-M4 @ 180MHz is optimized for fixed-point operations,")
    print("but slower than PC for floating-point. INT8 quantization helps here.")
    
    # Best model recommendation
    if results:
        # Find best model that fits F446RE
        fitting = [r for r in results if r['model_size_kb'] < 400]
        if fitting:
            best = max(fitting, key=lambda x: x['accuracy'])
            print(f"\nRecommended for STM32 F446RE: {best['name']}")
            print(f"  Accuracy: {best['accuracy']:.2f}%")
            print(f"  Size: {best['model_size_kb']:.1f} KB")


if __name__ == "__main__":
    main()

