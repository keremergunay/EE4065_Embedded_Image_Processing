"""
EE4065 Homework 5 - Run All Scripts
Master script to train models, convert to TFLite, and extract weights for STM32

Usage: python run_all.py
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Parent HW_5 folder

def run_script(script_name, description):
    """Run a Python script and display status."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    
    script_path = os.path.join(SCRIPT_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with return code {result.returncode}")
        return False
    
    print(f"\n[OK] {script_name} completed successfully!")
    return True


def main():
    print("=" * 70)
    print("EE4065 HOMEWORK 5 - EMBEDDED MACHINE LEARNING")
    print("STM32 Nucleo-F446RE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Train the KWS (Keyword Spotting) MLP model")
    print("  2. Train the HDR (Handwritten Digit Recognition) MLP model")
    print("  3. Convert both models to TFLite format")
    print("  4. Extract weights as C headers for STM32")
    print("\n" + "-" * 70)
    
    # Q1: Keyword Spotting
    if not run_script("Q1_keyword_spotting.py", "Q1: Keyword Spotting from Audio Signals"):
        print("\n[!] Q1 failed, but continuing to Q2...")
    
    # Q2: Handwritten Digit Recognition
    if not run_script("Q2_handwritten_digit_recognition.py", "Q2: Handwritten Digit Recognition"):
        print("\n[!] Q2 failed, but continuing to conversion...")
    
    # Convert to TFLite
    if not run_script("convert_to_tflite.py", "TFLite Conversion"):
        print("\n[!] TFLite conversion failed, but continuing to weight extraction...")
    
    # Extract weights for STM32
    if not run_script("extract_weights.py", "Weight Extraction for STM32"):
        print("\n[!] Weight extraction failed!")
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED!")
    print("=" * 70)
    
    models_dir = os.path.join(PROJECT_DIR, "Models")
    if os.path.exists(models_dir):
        print("\n[Trained Models]")
        for f in sorted(os.listdir(models_dir)):
            filepath = os.path.join(models_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {f} ({size/1024:.2f} KB)")
    
    results_dir = os.path.join(PROJECT_DIR, "Results")
    if os.path.exists(results_dir):
        print("\n[Results]")
        for f in sorted(os.listdir(results_dir)):
            print(f"  {f}")
    
    stm32_dir = os.path.join(PROJECT_DIR, "STM32")
    if os.path.exists(stm32_dir):
        print("\n[STM32 Files]")
        for f in sorted(os.listdir(stm32_dir)):
            filepath = os.path.join(stm32_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {f} ({size/1024:.2f} KB)")
    
    print("\n" + "-" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("""
1. Copy STM32/ files to your CubeIDE project:
   - main.cpp -> Core/Src/main.cpp
   - kws_weights.h -> Core/Inc/
   - hdr_weights.h -> Core/Inc/

2. Exclude usart.c from build (or use extern huart2)

3. Build and flash to STM32 Nucleo-F446RE

4. Run hw5_interactive.py to test:
   python hw5_interactive.py
""")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
