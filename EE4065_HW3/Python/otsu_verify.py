import serial
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# Settings
PORT = 'COM5'       
BAUD_RATE = 115200 
WIDTH = 160
HEIGHT = 120

def run_q1_grayscale(ser):
    print("\n--- Q1: Grayscale Otsu Starting ---")
    IMG_SIZE = WIDTH * HEIGHT
    
    # 1. Send Command ('1')
    ser.write(b'1')
    print("Command '1' sent. Waiting for result...")
    
    # 2. Read Data
    data = ser.read(IMG_SIZE)
    
    if len(data) == IMG_SIZE:
        print("Grayscale result received!")
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT, WIDTH))
        
        plt.figure("Q1 - Grayscale Otsu")
        plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
        plt.title("Question 1: Otsu Threshold (Grayscale)")
        plt.show()
    else:
        print(f"Error: Missing data ({len(data)} bytes received, expected {IMG_SIZE})")

def run_q2_color(ser):
    print("\n--- Q2: Color Otsu Starting ---")
    IMG_SIZE = WIDTH * HEIGHT * 3
    
    # 1. Prepare Image
    try:
        # Ensure the file path is correct on your system
        original = Image.open(r"D:\Projects\Embedded\PC_Python\mandrill.tiff").convert('RGB')
        resized = original.resize((WIDTH, HEIGHT))
        raw_bytes = np.array(resized).tobytes()
    except FileNotFoundError:
        print("ERROR: 'mandrill.tiff' file not found at the specified path!")
        return

    # 2. Send Load Command ('2')
    ser.write(b'2')
    print("Command '2' sent. Waiting for confirmation ('A')...")
    
    if ser.read(1) != b'A':
        print("Confirmation not received!")
        return
        
    # 3. Send Image Data
    print(f"Image is being loaded ({len(raw_bytes)} bytes)...")
    ser.write(raw_bytes)
    
    # 4. Wait for Processing Completion ('D')
    print("STM32 is processing...")
    if ser.read(1) != b'D':
        print("Processing timed out or error occurred!")
        return
    print("Processing completed.")
    
    # 5. Request Result ('3')
    ser.write(b'3')
    print("Result is being downloaded...")
    
    data = ser.read(IMG_SIZE)
    
    if len(data) == IMG_SIZE:
        print("Color result received!")
        img_array = np.frombuffer(data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        
        plt.figure("Q2 - Color Otsu (RGB)", figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(resized); plt.title("Original")
        plt.subplot(1,2,2); plt.imshow(img_array); plt.title("Processed (Otsu)")
        plt.show()
    else:
        print(f"Error: Missing data ({len(data)} bytes)")

def run_q3_morphology(ser):
    print("\n--- Q3: Morphological Operations ---")
    IMG_SIZE = WIDTH * HEIGHT
    
    # IMPORTANT: We must fetch Q1 result first to have a "Before" image for comparison
    # and to ensure STM32 has the binary image in memory.
    print("Requesting Q1 Otsu result first (Required for reference)...")
    ser.write(b'1')
    data_q1 = ser.read(IMG_SIZE)
    
    if len(data_q1) != IMG_SIZE:
        print("Error: Could not retrieve Q1 result. Cannot proceed.")
        return
        
    img_q1 = np.frombuffer(data_q1, dtype=np.uint8).reshape((HEIGHT, WIDTH))
    
    # --- Sub-Menu for Morphology ---
    print("\nSelect Operation:")
    print("4. Erosion")
    print("5. Dilation")
    print("6. Opening (Erosion -> Dilation)")
    print("7. Closing (Dilation -> Erosion)")
    print("b. Back")
    
    choice = input("Your choice: ")
    
    if choice not in ['4', '5', '6', '7']:
        if choice != 'b':
             print("Invalid choice.")
        return

    # Map choice to readable name
    op_names = {
        '4': "Erosion",
        '5': "Dilation",
        '6': "Opening",
        '7': "Closing"
    }
    op_name = op_names[choice]

    # Send Command
    print(f"Sending command '{choice}' ({op_name}). Waiting for result...")
    ser.write(choice.encode('ascii'))
    
    # Read Result
    data_q3 = ser.read(IMG_SIZE)

    if len(data_q3) == IMG_SIZE:
        print(f"{op_name} result received!")
        result_array = np.frombuffer(data_q3, dtype=np.uint8).reshape((HEIGHT, WIDTH))
        
        # Display Comparison
        plt.figure(f"Q3 - {op_name}", figsize=(10, 5))
        

        plt.plot()
        plt.imshow(result_array, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Result: {op_name}")
        plt.axis('off')
        
        plt.show()
    else:
        print(f"Error: Missing data for {op_name} ({len(data_q3)} bytes)")

def main():
    ser = None
    try:
        # Open Serial Port
        ser = serial.Serial(PORT, BAUD_RATE, timeout=10)
        time.sleep(2) # Wait for connection to stabilize
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"Connected to {PORT} at {BAUD_RATE} baud.")
        
        while True:
            print("\n" + "="*40)
            print("EE4065 - HW3 Menu")
            print("1. Run Question 1 (Grayscale Otsu)")
            print("2. Run Question 2 (Color Otsu)")
            print("3. Run Question 3 (Morphological Ops)")
            print("q. Exit")
            print("="*40)
            choice = input("Your choice: ")
            
            if choice == '1':
                run_q1_grayscale(ser)
            elif choice == '2':
                run_q2_color(ser)
            elif choice == '3':
                run_q3_morphology(ser)
            elif choice == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
            
            # Clear buffer after every operation to ensure clean state
            ser.reset_input_buffer()

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Port closed.")

if __name__ == "__main__":
    main()