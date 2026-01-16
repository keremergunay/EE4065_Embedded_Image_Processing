"""
EE4065 Homework 5 - Interactive PC-STM32 Test Script
Combines Q1 (Keyword Spotting) and Q2 (Handwritten Digit Recognition)

Similar interface to HW_3 otsu_verify.py
"""

import serial
import numpy as np
import scipy.signal as sig
import cv2
import os
import sys
import time
import struct

# Add path for imports
sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf
from tensorflow import keras

# ============================================================================
# Configuration - UPDATE THIS
# ============================================================================
PORT = 'COM5'          # Your STM32's COM port
BAUD_RATE = 115200
TIMEOUT = 5

# Protocol
SYNC_BYTE = 0xAA
ACK_BYTE = 0x55

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
FSDD_PATH = os.path.join(PROJECT_DIR, "..", 
    "Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main",
    "Data", "FSDD")

# MFCC Parameters
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13

# ============================================================================
# Global Variables
# ============================================================================
kws_interpreter = None
hdr_interpreter = None
mnist_images = None
mnist_labels = None

# ============================================================================
# Initialization
# ============================================================================

def init_models():
    """Load Keras models for PC comparison."""
    global kws_interpreter, hdr_interpreter, mnist_images, mnist_labels
    
    print("Loading models...")
    
    # KWS Model
    kws_path = os.path.join(MODELS_DIR, "kws_mlp.h5")
    if os.path.exists(kws_path):
        kws_interpreter = keras.models.load_model(kws_path, compile=False)
        print("  [OK] KWS model loaded")
    else:
        print(f"  [WARN] KWS model not found at {kws_path}")
    
    # HDR Model
    hdr_path = os.path.join(MODELS_DIR, "hdr_mlp.h5")
    if os.path.exists(hdr_path):
        hdr_interpreter = keras.models.load_model(hdr_path, compile=False)
        print("  [OK] HDR model loaded")
    else:
        print(f"  [WARN] HDR model not found at {hdr_path}")
    
    # Load MNIST
    print("Loading MNIST dataset...")
    (_, _), (mnist_images, mnist_labels) = keras.datasets.mnist.load_data()
    print(f"  [OK] {len(mnist_images)} test images loaded")

# ============================================================================
# Feature Extraction
# ============================================================================

def extract_mfcc(wav_path):
    """Extract MFCC features from audio file."""
    from mfcc_func import create_mfcc_features
    window = sig.get_window("hamming", FFTSize)
    features, _ = create_mfcc_features(
        [wav_path], FFTSize, sample_rate, 
        numOfMelFilters, numOfDctOutputs, window
    )
    return features[0].astype(np.float32)

def extract_hu_moments(image):
    """Extract Hu Moments from image."""
    moments = cv2.moments(image, True)
    hu = cv2.HuMoments(moments).flatten()
    return hu.astype(np.float32)

# ============================================================================
# PC Inference
# ============================================================================

def pc_inference_kws(features):
    """Run KWS inference on PC using Keras model."""
    if kws_interpreter is None:
        return None
    preds = kws_interpreter.predict(features.reshape(1, 26).astype(np.float32), verbose=0)
    return preds[0]

def pc_inference_hdr(hu_moments):
    """Run HDR inference on PC using Keras model."""
    if hdr_interpreter is None:
        return None
    preds = hdr_interpreter.predict(hu_moments.reshape(1, 7).astype(np.float32), verbose=0)
    return preds[0]

# ============================================================================
# STM32 Communication
# ============================================================================

def send_mode_to_stm32(ser, mode):
    """Send mode selection to STM32 ('1' for KWS, '2' for HDR)."""
    ser.write(mode.encode('ascii'))
    time.sleep(0.1)
    # Read any response
    response = ""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
    return response

def send_features_get_prediction(ser, features, num_features):
    """Send features to STM32 and receive predictions."""
    try:
        # Clear buffer
        ser.reset_input_buffer()
        
        # Send sync byte
        ser.write(bytes([SYNC_BYTE]))
        
        # Wait for ACK (with timeout)
        start = time.time()
        while time.time() - start < 2:
            if ser.in_waiting > 0:
                ack = ser.read(1)
                if len(ack) > 0 and ack[0] == ACK_BYTE:
                    break
        else:
            return None, "Timeout waiting for ACK"
        
        # Send features
        data = features[:num_features].astype(np.float32).tobytes()
        ser.write(data)
        
        # Wait for predictions (10 floats = 40 bytes)
        start = time.time()
        response = b''
        while len(response) < 40 and time.time() - start < 2:
            if ser.in_waiting > 0:
                response += ser.read(ser.in_waiting)
            time.sleep(0.01)
        
        if len(response) < 40:
            return None, f"Incomplete response ({len(response)} bytes)"
        
        predictions = np.frombuffer(response[:40], dtype=np.float32)
        return predictions, None
        
    except Exception as e:
        return None, str(e)

# ============================================================================
# Menu Functions
# ============================================================================

def run_q1_kws(ser):
    """Q1: Keyword Spotting - Test with audio files."""
    print("\n" + "="*50)
    print("Q1: KEYWORD SPOTTING (KWS)")
    print("="*50)
    
    if not os.path.exists(FSDD_PATH):
        print(f"ERROR: FSDD dataset not found at {FSDD_PATH}")
        return
    
    # Send mode to STM32
    print("\nSending mode '1' to STM32...")
    response = send_mode_to_stm32(ser, '1')
    if response:
        print(f"STM32: {response}")
    time.sleep(0.5)
    
    # Get test files
    test_files = sorted([f for f in os.listdir(FSDD_PATH) 
                        if f.endswith('.wav') and "yweweler" in f])
    
    print(f"\nAvailable test files: {len(test_files)}")
    print("\nSelect option:")
    print("  a. Test ALL files")
    print("  r. Test RANDOM 5 files")
    print("  s. Select SPECIFIC digit (0-9)")
    print("  b. Back to main menu")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'b':
        return
    elif choice == 'a':
        files_to_test = test_files[:20]  # Limit to 20
    elif choice == 'r':
        files_to_test = np.random.choice(test_files, min(5, len(test_files)), replace=False)
    elif choice == 's':
        digit = input("Enter digit (0-9): ").strip()
        if digit not in '0123456789':
            print("Invalid digit!")
            return
        files_to_test = [f for f in test_files if f.startswith(f"{digit}_")][:5]
    else:
        print("Invalid choice!")
        return
    
    print(f"\nTesting {len(files_to_test)} files...")
    print("-" * 50)
    
    correct_pc = 0
    correct_mcu = 0
    total = 0
    
    for wav_file in files_to_test:
        wav_path = os.path.join(FSDD_PATH, wav_file)
        true_digit = int(wav_file.split("_")[0])
        
        print(f"\n{wav_file}")
        print(f"  True digit: {true_digit}")
        
        # Extract features
        try:
            features = extract_mfcc(wav_path)
        except Exception as e:
            print(f"  Feature extraction error: {e}")
            continue
        
        # PC inference
        pc_preds = pc_inference_kws(features)
        pc_digit = np.argmax(pc_preds)
        pc_conf = pc_preds[pc_digit] * 100
        
        # MCU inference
        mcu_preds, error = send_features_get_prediction(ser, features, 26)
        
        if mcu_preds is not None:
            mcu_digit = np.argmax(mcu_preds)
            mcu_conf = mcu_preds[mcu_digit] * 100
            
            pc_ok = "✓" if pc_digit == true_digit else "✗"
            mcu_ok = "✓" if mcu_digit == true_digit else "✗"
            
            print(f"  PC:  {pc_digit} ({pc_conf:5.1f}%) {pc_ok}")
            print(f"  MCU: {mcu_digit} ({mcu_conf:5.1f}%) {mcu_ok}")
            
            if pc_digit == true_digit:
                correct_pc += 1
            if mcu_digit == true_digit:
                correct_mcu += 1
            total += 1
        else:
            print(f"  MCU Error: {error}")
    
    # Summary
    if total > 0:
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"PC Accuracy:  {correct_pc}/{total} = {100*correct_pc/total:.1f}%")
        print(f"MCU Accuracy: {correct_mcu}/{total} = {100*correct_mcu/total:.1f}%")
    
    input("\nPress Enter to continue...")

def run_q2_hdr(ser):
    """Q2: Handwritten Digit Recognition - Test with MNIST."""
    print("\n" + "="*50)
    print("Q2: HANDWRITTEN DIGIT RECOGNITION (HDR)")
    print("="*50)
    
    # Send mode to STM32
    print("\nSending mode '2' to STM32...")
    response = send_mode_to_stm32(ser, '2')
    if response:
        print(f"STM32: {response}")
    time.sleep(0.5)
    
    print("\nSelect option:")
    print("  r. Test RANDOM 10 samples")
    print("  s. Select SPECIFIC digit (0-9)")
    print("  n. Enter NUMBER of random tests")
    print("  b. Back to main menu")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'b':
        return
    elif choice == 'r':
        indices = np.random.choice(len(mnist_images), 10, replace=False)
    elif choice == 's':
        digit = input("Enter digit (0-9): ").strip()
        if digit not in '0123456789':
            print("Invalid digit!")
            return
        digit = int(digit)
        digit_indices = np.where(mnist_labels == digit)[0]
        indices = np.random.choice(digit_indices, min(5, len(digit_indices)), replace=False)
    elif choice == 'n':
        try:
            n = int(input("How many tests? ").strip())
            n = min(max(1, n), 50)  # Limit 1-50
            indices = np.random.choice(len(mnist_images), n, replace=False)
        except:
            print("Invalid number!")
            return
    else:
        print("Invalid choice!")
        return
    
    print(f"\nTesting {len(indices)} samples...")
    print("-" * 50)
    
    correct_pc = 0
    correct_mcu = 0
    total = 0
    
    for i, idx in enumerate(indices):
        image = mnist_images[idx]
        true_label = mnist_labels[idx]
        
        print(f"\nSample {i+1}/{len(indices)} (Index: {idx})")
        print(f"  True digit: {true_label}")
        
        # Extract Hu Moments
        hu = extract_hu_moments(image)
        
        # PC inference
        pc_preds = pc_inference_hdr(hu)
        pc_digit = np.argmax(pc_preds)
        pc_conf = pc_preds[pc_digit] * 100
        
        # MCU inference
        mcu_preds, error = send_features_get_prediction(ser, hu, 7)
        
        if mcu_preds is not None:
            mcu_digit = np.argmax(mcu_preds)
            mcu_conf = mcu_preds[mcu_digit] * 100
            
            pc_ok = "✓" if pc_digit == true_label else "✗"
            mcu_ok = "✓" if mcu_digit == true_label else "✗"
            
            print(f"  PC:  {pc_digit} ({pc_conf:5.1f}%) {pc_ok}")
            print(f"  MCU: {mcu_digit} ({mcu_conf:5.1f}%) {mcu_ok}")
            
            if pc_digit == true_label:
                correct_pc += 1
            if mcu_digit == true_label:
                correct_mcu += 1
            total += 1
        else:
            print(f"  MCU Error: {error}")
    
    # Summary
    if total > 0:
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"PC Accuracy:  {correct_pc}/{total} = {100*correct_pc/total:.1f}%")
        print(f"MCU Accuracy: {correct_mcu}/{total} = {100*correct_mcu/total:.1f}%")
    
    input("\nPress Enter to continue...")

def show_stm32_info(ser):
    """Request info from STM32."""
    print("\nRequesting info from STM32...")
    ser.write(b'i')
    time.sleep(0.3)
    
    response = ""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        time.sleep(0.1)
    
    if response:
        print(response)
    else:
        print("No response from STM32")
    
    input("\nPress Enter to continue...")

def show_stm32_menu(ser):
    """Show menu on STM32."""
    ser.write(b'?')
    time.sleep(0.3)
    
    response = ""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
        time.sleep(0.1)
    
    if response:
        print(response)

# ============================================================================
# Main
# ============================================================================

def main():
    ser = None
    
    try:
        # Initialize models
        init_models()
        
        # Open serial port
        print(f"\nConnecting to {PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for STM32 to reset
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connected to {PORT}!")
        
        # Read startup message
        time.sleep(0.5)
        while ser.in_waiting:
            print(ser.read(ser.in_waiting).decode('utf-8', errors='ignore'), end='')
        
        # Main menu loop
        while True:
            print("\n" + "="*50)
            print("EE4065 - HW5: Embedded Machine Learning")
            print("="*50)
            print("1. Run Q1 (Keyword Spotting from Audio)")
            print("2. Run Q2 (Handwritten Digit Recognition)")
            print("i. Show STM32 Info")
            print("m. Show STM32 Menu")
            print("q. Exit")
            print("="*50)
            
            choice = input("Your choice: ").strip().lower()
            
            if choice == '1':
                run_q1_kws(ser)
            elif choice == '2':
                run_q2_hdr(ser)
            elif choice == 'i':
                show_stm32_info(ser)
            elif choice == 'm':
                show_stm32_menu(ser)
            elif choice == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Try again.")
            
            # Clear buffer
            ser.reset_input_buffer()
    
    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Make sure:")
        print(f"  1. STM32 is connected to {PORT}")
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
            print("Port closed.")

if __name__ == "__main__":
    main()

