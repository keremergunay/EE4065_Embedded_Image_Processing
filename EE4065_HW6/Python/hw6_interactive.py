"""
EE4065 Homework 6 - Interactive PC-STM32 Test Script
CNN-based Handwritten Digit Recognition using TensorFlow Lite Micro

Compatible Models for STM32 Nucleo-F446RE (512KB Flash, 128KB RAM):
  - SqueezeNetMini:    54.2 KB  (SMALLEST - Recommended)
  - ResNet8:           96.3 KB
  - MobileNetV2Mini:  106.5 KB
  - EfficientNetMini: 108.5 KB
  - ResNet14:         201.7 KB  (May have tight RAM)

Note: Each model must be tested one at a time due to Flash constraints.
      Change the model in STM32 project and reflash to test different models.
"""

import os
import sys

# Disable XNNPACK delegate (must be set BEFORE importing TensorFlow)
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'

import serial
import serial.tools.list_ports
import numpy as np
import cv2
import time
import struct

import tensorflow as tf
from tensorflow import keras

# ============================================================================
# Configuration
# ============================================================================
PORT = 'COM5'          # Your STM32's COM port (update as needed)
BAUD_RATE = 115200
TIMEOUT = 10

# Protocol bytes
SYNC_BYTE = 0xAA
ACK_BYTE = 0x55
CMD_INFERENCE = 0x01
CMD_INFO = 0x02
CMD_READY = 0x03

# Image parameters (matching STM32 model input)
IMAGE_SIZE = 32
NUM_CLASSES = 10

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_DIR = os.path.join(SCRIPT_DIR, "tflite_models")

# ============================================================================
# Global Variables
# ============================================================================
mnist_images = None
mnist_labels = None
tflite_interpreters = {}

# ============================================================================
# Model Information
# ============================================================================
MODEL_INFO = {
    'SqueezeNetMini': {'size_kb': 54.2, 'fits_f446': True, 'recommended': True},
    'ResNet8': {'size_kb': 96.3, 'fits_f446': True, 'recommended': False},
    'MobileNetV2Mini': {'size_kb': 106.5, 'fits_f446': True, 'recommended': False},
    'EfficientNetMini': {'size_kb': 108.5, 'fits_f446': True, 'recommended': False},
    'ResNet14': {'size_kb': 201.7, 'fits_f446': True, 'recommended': False},
    'ResNet20': {'size_kb': 307.2, 'fits_f446': False, 'recommended': False},
    'MobileNetV2': {'size_kb': 492.1, 'fits_f446': False, 'recommended': False},
    'SqueezeNet': {'size_kb': 805.8, 'fits_f446': False, 'recommended': False},
    'EfficientNet': {'size_kb': 767.1, 'fits_f446': False, 'recommended': False},
    'ShuffleNetMini': {'size_kb': 253.2, 'fits_f446': False, 'recommended': False},
    'ShuffleNet': {'size_kb': 1159.0, 'fits_f446': False, 'recommended': False},
}

# ============================================================================
# Initialization
# ============================================================================

def init_models():
    """Load MNIST dataset and TFLite interpreters for PC comparison."""
    global mnist_images, mnist_labels, tflite_interpreters
    
    print("Loading MNIST dataset...")
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # Preprocess: resize to 32x32
    mnist_images = []
    for img in test_images:
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        mnist_images.append(img_resized)
    mnist_images = np.array(mnist_images)
    mnist_labels = test_labels
    print(f"  [OK] {len(mnist_images)} test images loaded")
    
    # Load TFLite interpreters for compatible models
    print("\nLoading TFLite models for PC comparison...")
    for model_name, info in MODEL_INFO.items():
        if info['fits_f446']:
            tflite_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")
            if os.path.exists(tflite_path):
                try:
                    # Use empty list to disable all delegates including XNNPACK
                    interpreter = tf.lite.Interpreter(
                        model_path=tflite_path,
                        num_threads=1,
                        experimental_delegates=[]  # Empty list = no delegates
                    )
                    interpreter.allocate_tensors()
                    tflite_interpreters[model_name] = interpreter
                    print(f"  [OK] {model_name} ({info['size_kb']:.1f} KB)")
                except Exception as e:
                    print(f"  [WARN] {model_name}: {e}")
            else:
                print(f"  [SKIP] {model_name}: not found")

def list_available_ports():
    """List available COM ports."""
    ports = serial.tools.list_ports.comports()
    print("\nAvailable COM ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port.device}: {port.description}")
    return ports

# ============================================================================
# PC Inference using TFLite
# ============================================================================

def pc_inference(image, model_name='SqueezeNetMini'):
    """Run inference on PC using TFLite interpreter."""
    if model_name not in tflite_interpreters:
        return None, None
    
    interpreter = tflite_interpreters[model_name]
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Prepare image: expand to 3 channels, normalize
    img = image.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Quantize if INT8 model
    if input_details['dtype'] == np.uint8:
        scale, zero_point = input_details['quantization']
        img = (img / scale + zero_point).astype(np.uint8)
    
    interpreter.set_tensor(input_details['index'], img)
    
    try:
        start = time.perf_counter()
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
    except RuntimeError as e:
        # XNNPACK delegate may fail for some models - skip PC comparison
        print(f"  [PC inference failed: {model_name} - XNNPACK incompatible]")
        return None, None
        
    output = interpreter.get_tensor(output_details['index'])
    
    # Dequantize if INT8
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    return output[0], elapsed

# ============================================================================
# STM32 Communication
# ============================================================================

def send_image_get_prediction(ser, image):
    """Send image to STM32 and receive prediction."""
    try:
        # Clear buffer
        ser.reset_input_buffer()
        
        # Prepare image: grayscale uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        image_flat = image.flatten()
        
        # Send sync byte
        ser.write(bytes([SYNC_BYTE]))
        
        # Wait for ACK
        start = time.time()
        while time.time() - start < 2:
            if ser.in_waiting > 0:
                ack = ser.read(1)
                if len(ack) > 0 and ack[0] == ACK_BYTE:
                    break
        else:
            return None, None, "Timeout waiting for ACK"
        
        # Send image data (32*32 = 1024 bytes)
        ser.write(image_flat.tobytes())
        
        # Wait for predictions (10 floats = 40 bytes for float, or 10 bytes for uint8)
        start = time.time()
        response = b''
        expected_size = NUM_CLASSES  # uint8 output from quantized model
        
        while len(response) < expected_size and time.time() - start < 5:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting)
            time.sleep(0.01)
        
        if len(response) < expected_size:
            return None, None, f"Incomplete response ({len(response)} bytes)"
        
        # Parse as uint8 (quantized output)
        predictions = np.frombuffer(response[:expected_size], dtype=np.uint8)
        
        # Read inference time (4 bytes, uint32)
        time_data = b''
        start = time.time()
        while len(time_data) < 4 and time.time() - start < 1:
            if ser.in_waiting > 0:
                time_data += ser.read(min(4 - len(time_data), ser.in_waiting))
            time.sleep(0.01)
        
        if len(time_data) >= 4:
            inference_time = struct.unpack('<I', time_data[:4])[0]
        else:
            inference_time = 0
        
        return predictions, inference_time, None
        
    except Exception as e:
        return None, None, str(e)

def request_stm32_info(ser):
    """Request info from STM32."""
    ser.reset_input_buffer()
    ser.write(b'i')
    time.sleep(0.5)
    
    response = ""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        time.sleep(0.1)
    
    return response

def request_stm32_menu(ser):
    """Request menu from STM32."""
    ser.reset_input_buffer()
    ser.write(b'?')
    time.sleep(0.5)
    
    response = ""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        time.sleep(0.1)
    
    return response

# ============================================================================
# Display Functions
# ============================================================================

def display_image_ascii(image):
    """Display image as ASCII art."""
    print("\n  Image preview:")
    for row in range(0, IMAGE_SIZE, 2):
        line = "  "
        for col in range(0, IMAGE_SIZE, 1):
            pixel = image[row, col] if row < IMAGE_SIZE else 0
            if pixel > 200:
                line += "██"
            elif pixel > 150:
                line += "▓▓"
            elif pixel > 100:
                line += "▒▒"
            elif pixel > 50:
                line += "░░"
            else:
                line += "  "
        print(line)

def display_predictions(predictions, label=None):
    """Display prediction bar chart."""
    predicted = np.argmax(predictions)
    max_val = max(predictions)
    
    print("\n  Predictions:")
    for i in range(NUM_CLASSES):
        val = predictions[i]
        bar_len = int(val / max_val * 20) if max_val > 0 else 0
        bar = '█' * bar_len
        marker = " ← PREDICTED" if i == predicted else ""
        if label is not None and i == label:
            marker += " (TRUE)"
        print(f"    {i}: {val:3d} {bar}{marker}")
    
    return predicted

# ============================================================================
# Test Functions
# ============================================================================

def run_single_test(ser, model_name='SqueezeNetMini'):
    """Run single test with random image."""
    idx = np.random.randint(0, len(mnist_images))
    image = mnist_images[idx]
    true_label = mnist_labels[idx]
    
    print(f"\n{'='*50}")
    print(f"Test Sample (Index: {idx})")
    print(f"{'='*50}")
    print(f"True label: {true_label}")
    
    display_image_ascii(image)
    
    # PC inference
    pc_preds, pc_time = pc_inference(image, model_name)
    if pc_preds is not None:
        # Convert to uint8-like scale for comparison
        pc_preds_scaled = (pc_preds * 255).astype(np.uint8)
        pc_pred = np.argmax(pc_preds_scaled)
        print(f"\n  PC ({model_name}): Predicted {pc_pred} ({pc_time:.2f}ms)")
    else:
        pc_pred = -1
        print(f"\n  PC ({model_name}): Skipped (delegate error)")
    
    # MCU inference
    mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
    
    if mcu_preds is not None:
        mcu_pred = display_predictions(mcu_preds, true_label)
        print(f"\n  MCU: Predicted {mcu_pred} (Inference time: {mcu_time}ms)")
        
        # Compare
        mcu_correct = "✓" if mcu_pred == true_label else "✗"
        print(f"\n  Result: MCU={mcu_pred} {mcu_correct}  (True={true_label})")
    else:
        print(f"\n  MCU Error: {error}")
        mcu_pred = -1
    
    return true_label, pc_pred, mcu_pred

def run_batch_test(ser, num_samples=10, model_name='SqueezeNetMini'):
    """Run batch test."""
    print(f"\n{'='*50}")
    print(f"Batch Test: {num_samples} samples")
    print(f"{'='*50}")
    
    indices = np.random.choice(len(mnist_images), num_samples, replace=False)
    
    correct_pc = 0
    correct_mcu = 0
    total = 0
    total_pc_time = 0
    total_mcu_time = 0
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        # PC inference
        pc_preds, pc_time = pc_inference(image, model_name)
        if pc_preds is not None:
            pc_pred = np.argmax(pc_preds)
            total_pc_time += pc_time
            if pc_pred == true_label:
                correct_pc += 1
        else:
            pc_pred = -1
        
        # MCU inference
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            total_mcu_time += mcu_time
            if mcu_pred == true_label:
                correct_mcu += 1
            total += 1
            
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            
            if pc_pred >= 0:
                pc_ok = "✓" if pc_pred == true_label else "✗"
                print(f"  [{i+1:3d}/{num_samples}] True={true_label}  PC={pc_pred}{pc_ok}  MCU={mcu_pred}{mcu_ok}  ({mcu_time}ms)")
            else:
                print(f"  [{i+1:3d}/{num_samples}] True={true_label}  PC=N/A  MCU={mcu_pred}{mcu_ok}  ({mcu_time}ms)")
        else:
            print(f"  [{i+1:3d}/{num_samples}] ERROR: {error}")
    
    # Summary
    if total > 0:
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        if total_pc_time > 0:
            print(f"  PC Accuracy:  {correct_pc}/{total} = {100*correct_pc/total:.1f}%")
            print(f"  Avg PC Time:  {total_pc_time/total:.2f}ms")
        else:
            print(f"  PC: Skipped (delegate error)")
        print(f"  MCU Accuracy: {correct_mcu}/{total} = {100*correct_mcu/total:.1f}%")
        print(f"  Avg MCU Time: {total_mcu_time/total:.1f}ms")

def run_digit_test(ser, digit, model_name='SqueezeNetMini'):
    """Test specific digit."""
    print(f"\n{'='*50}")
    print(f"Testing Digit: {digit}")
    print(f"{'='*50}")
    
    # Find samples of this digit
    digit_indices = np.where(mnist_labels == digit)[0]
    indices = np.random.choice(digit_indices, min(5, len(digit_indices)), replace=False)
    
    run_batch_test_indices(ser, indices, model_name)

def run_batch_test_indices(ser, indices, model_name='SqueezeNetMini'):
    """Run batch test on specific indices."""
    correct_mcu = 0
    total = 0
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        mcu_preds, mcu_time, error = send_image_get_prediction(ser, image)
        
        if mcu_preds is not None:
            mcu_pred = np.argmax(mcu_preds)
            if mcu_pred == true_label:
                correct_mcu += 1
            total += 1
            
            mcu_ok = "✓" if mcu_pred == true_label else "✗"
            print(f"  [{i+1:3d}] True={true_label}  MCU={mcu_pred} {mcu_ok}  ({mcu_time}ms)")
        else:
            print(f"  [{i+1:3d}] ERROR: {error}")
    
    if total > 0:
        print(f"\n  Accuracy: {correct_mcu}/{total} = {100*correct_mcu/total:.1f}%")

# ============================================================================
# Menu Functions
# ============================================================================

def show_model_compatibility():
    """Show which models are compatible with F446RE."""
    print("\n" + "="*60)
    print("Model Compatibility for STM32 Nucleo-F446RE")
    print("(512KB Flash, 128KB RAM)")
    print("="*60)
    print(f"{'Model':<20} {'Size':<12} {'Compatible':<12} {'Notes'}")
    print("-"*60)
    
    for name, info in sorted(MODEL_INFO.items(), key=lambda x: x[1]['size_kb']):
        size_str = f"{info['size_kb']:.1f} KB"
        compat = "✓ YES" if info['fits_f446'] else "✗ NO"
        notes = "← RECOMMENDED" if info['recommended'] else ""
        print(f"{name:<20} {size_str:<12} {compat:<12} {notes}")
    
    print("-"*60)
    print("Note: Only ONE model can be loaded at a time due to Flash size.")
    print("      Change model in STM32 project and reflash to test others.")

def interactive_menu(ser):
    """Interactive menu loop."""
    current_model = 'SqueezeNetMini'  # Default model (should match STM32)
    
    while True:
        print("\n" + "="*60)
        print("EE4065 HW6 - CNN Handwritten Digit Recognition")
        print("STM32 Nucleo-F446RE + TensorFlow Lite Micro")
        print("="*60)
        print(f"Current model (PC comparison): {current_model}")
        print("-"*60)
        print("  1. Single image test (random)")
        print("  2. Batch test (10 random images)")
        print("  3. Batch test (custom count)")
        print("  4. Test specific digit (0-9)")
        print("  5. Show model compatibility")
        print("  6. Change PC comparison model")
        print("  i. Show STM32 info")
        print("  m. Show STM32 menu")
        print("  q. Quit")
        print("="*60)
        
        choice = input("Your choice: ").strip().lower()
        
        if choice == '1':
            run_single_test(ser, current_model)
            input("\nPress Enter to continue...")
        
        elif choice == '2':
            run_batch_test(ser, 10, current_model)
            input("\nPress Enter to continue...")
        
        elif choice == '3':
            try:
                n = int(input("How many samples? (1-100): ").strip())
                n = max(1, min(100, n))
                run_batch_test(ser, n, current_model)
            except:
                print("Invalid number!")
            input("\nPress Enter to continue...")
        
        elif choice == '4':
            digit = input("Enter digit (0-9): ").strip()
            if digit in '0123456789':
                run_digit_test(ser, int(digit), current_model)
            else:
                print("Invalid digit!")
            input("\nPress Enter to continue...")
        
        elif choice == '5':
            show_model_compatibility()
            input("\nPress Enter to continue...")
        
        elif choice == '6':
            print("\nAvailable models (for PC comparison):")
            models = list(tflite_interpreters.keys())
            for i, name in enumerate(models):
                marker = " (current)" if name == current_model else ""
                print(f"  {i+1}. {name}{marker}")
            try:
                idx = int(input("Select model: ").strip()) - 1
                if 0 <= idx < len(models):
                    current_model = models[idx]
                    print(f"Changed to: {current_model}")
            except:
                print("Invalid selection!")
        
        elif choice == 'i':
            response = request_stm32_info(ser)
            print(response if response else "No response from STM32")
            input("\nPress Enter to continue...")
        
        elif choice == 'm':
            response = request_stm32_menu(ser)
            print(response if response else "No response from STM32")
            input("\nPress Enter to continue...")
        
        elif choice == 'q':
            break
        
        else:
            print("Invalid choice. Try again.")
        
        # Clear buffer
        ser.reset_input_buffer()

# ============================================================================
# Main
# ============================================================================

def main():
    ser = None
    
    try:
        # Initialize models
        init_models()
        
        # Show model compatibility
        show_model_compatibility()
        
        # List available ports
        ports = list_available_ports()
        
        if not ports:
            print("\nNo COM ports found!")
            print("Make sure the STM32 Nucleo board is connected via USB.")
            return
        
        # Select port
        print(f"\nDefault port: {PORT}")
        print("Enter COM port (or press Enter for default, 'q' to quit):")
        port_input = input("> ").strip()
        
        if port_input.lower() == 'q':
            return
        
        if port_input:
            # Check if input is a number (index)
            try:
                idx = int(port_input)
                if 0 <= idx < len(ports):
                    port = ports[idx].device
                else:
                    port = port_input
            except ValueError:
                port = port_input
        else:
            port = PORT
        
        # Connect
        print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
        ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for STM32 to reset
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connected to {port}!")
        
        # Read startup message
        time.sleep(0.5)
        while ser.in_waiting:
            print(ser.read(ser.in_waiting).decode('utf-8', errors='ignore'), end='')
        
        # Interactive menu
        interactive_menu(ser)
    
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Make sure:")
        print(f"  1. STM32 is connected")
        print("  2. No other program is using the port")
        print("  3. The correct firmware is flashed")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser and ser.is_open:
            ser.close()
            print("\nPort closed.")

if __name__ == "__main__":
    main()

