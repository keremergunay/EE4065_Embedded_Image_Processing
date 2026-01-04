"""
EE 4065 - Embedded Digital Image Processing - Homework 6
PC Serial Communication Script for STM32 Nucleo-F446RE

This script handles:
1. Sending MNIST images to STM32 over UART
2. Receiving classification results from STM32
3. Displaying and logging results

Target: STM32 Nucleo-F446RE
Communication: Virtual COM Port (ST-Link USB)
"""

import os
import sys
import time
import struct
import numpy as np
import serial
import serial.tools.list_ports
from tensorflow import keras
import cv2

# Configuration
BAUD_RATE = 115200  # F446RE can handle high baud rates
IMAGE_SIZE = 32
IMAGE_CHANNELS = 1  # Grayscale for MNIST (can also use 3 for RGB)
NUM_CLASSES = 10

# Protocol constants
HEADER_PC_TO_STM = b'PC2S'  # PC sends to STM32
HEADER_STM_TO_PC = b'S2PC'  # STM32 sends to PC
CMD_SEND_IMAGE = 0x01
CMD_RECV_RESULT = 0x02
CMD_READY = 0x03


class STM32SerialComm:
    """Serial communication handler for STM32 Nucleo-F446RE"""
    
    def __init__(self, port=None, baud_rate=BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.serial = None
    
    def list_ports(self):
        """List available COM ports"""
        ports = serial.tools.list_ports.comports()
        print("\nAvailable COM ports:")
        for i, port in enumerate(ports):
            print(f"  [{i}] {port.device}: {port.description}")
        return ports
    
    def connect(self, port=None):
        """Connect to STM32"""
        if port:
            self.port = port
        
        if not self.port:
            # Try to auto-detect STM32
            ports = self.list_ports()
            for p in ports:
                if 'STM' in p.description.upper() or 'ST-LINK' in p.description.upper():
                    self.port = p.device
                    break
            
            if not self.port and ports:
                self.port = ports[0].device
        
        if not self.port:
            raise Exception("No COM port found!")
        
        print(f"\nConnecting to {self.port} at {self.baud_rate} baud...")
        self.serial = serial.Serial(self.port, self.baud_rate, timeout=5)
        time.sleep(0.5)  # Wait for connection to stabilize
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        print("Connected!")
        return True
    
    def disconnect(self):
        """Disconnect from STM32"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Disconnected")
    
    def send_image(self, image):
        """
        Send an image to STM32 for classification
        
        Args:
            image: numpy array of shape (32, 32) or (32, 32, 1)
        
        Returns:
            True if sent successfully
        """
        # Ensure correct shape
        if image.ndim == 3:
            image = image[:, :, 0]
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Flatten image
        image_data = image.flatten().tobytes()
        image_size = len(image_data)
        
        # Send header: 'PC2S' + command + size (4 bytes)
        header = HEADER_PC_TO_STM + bytes([CMD_SEND_IMAGE])
        header += struct.pack('<I', image_size)  # Little-endian uint32
        
        self.serial.write(header)
        self.serial.write(image_data)
        self.serial.flush()
        
        return True
    
    def receive_result(self, timeout=5):
        """
        Receive classification result from STM32
        
        Returns:
            Tuple of (predicted_class, confidence_scores) or None on error
        """
        start_time = time.time()
        
        # Wait for response header
        while time.time() - start_time < timeout:
            if self.serial.in_waiting >= 4:
                header = self.serial.read(4)
                if header == HEADER_STM_TO_PC:
                    break
        else:
            print("Timeout waiting for STM32 response")
            return None
        
        # Read command and data length
        cmd = self.serial.read(1)[0]
        data_len = struct.unpack('<I', self.serial.read(4))[0]
        
        if cmd != CMD_RECV_RESULT:
            print(f"Unexpected command: {cmd}")
            return None
        
        # Read result data
        result_data = self.serial.read(data_len)
        
        # Parse results (10 confidence scores as uint8)
        if len(result_data) >= NUM_CLASSES:
            scores = np.frombuffer(result_data[:NUM_CLASSES], dtype=np.uint8)
            predicted_class = np.argmax(scores)
            return predicted_class, scores
        
        return None
    
    def wait_for_ready(self, timeout=10):
        """Wait for STM32 to signal it's ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.serial.in_waiting >= 5:
                data = self.serial.read(5)
                if data[:4] == HEADER_STM_TO_PC and data[4] == CMD_READY:
                    print("STM32 is ready!")
                    return True
            time.sleep(0.1)
        return False


def load_test_images(num_samples=10):
    """Load test images from MNIST dataset"""
    (_, _), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    # Select random samples
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    samples = test_images[indices]
    labels = test_labels[indices]
    
    # Resize to 32x32
    samples_resized = []
    for img in samples:
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        samples_resized.append(img_resized)
    
    return np.array(samples_resized), labels


def load_custom_image(image_path):
    """Load and preprocess a custom image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Resize to 32x32
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Invert if needed (MNIST has white digits on black background)
    if np.mean(img) > 127:
        img = 255 - img
    
    return img


def interactive_test(comm):
    """Interactive testing mode"""
    print("\n" + "=" * 50)
    print("Interactive Testing Mode")
    print("=" * 50)
    print("Commands:")
    print("  'r' or Enter - Send random MNIST image")
    print("  'f <path>' - Send custom image file")
    print("  'q' - Quit")
    print("=" * 50)
    
    while True:
        try:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == '' or cmd == 'r':
                # Send random MNIST image
                images, labels = load_test_images(1)
                image = images[0]
                true_label = labels[0]
                
                print(f"Sending MNIST digit (true label: {true_label})...")
                
            elif cmd.startswith('f '):
                # Send custom image
                image_path = cmd[2:].strip()
                image = load_custom_image(image_path)
                true_label = None
                print(f"Sending custom image: {image_path}")
            else:
                print("Unknown command")
                continue
            
            # Display image
            display_image(image)
            
            # Send to STM32
            if comm.send_image(image):
                print("Image sent, waiting for result...")
                
                # Receive result
                result = comm.receive_result()
                
                if result:
                    predicted, scores = result
                    print(f"\nPrediction: {predicted}")
                    if true_label is not None:
                        correct = "✓" if predicted == true_label else "✗"
                        print(f"True label: {true_label} {correct}")
                    
                    print("\nConfidence scores:")
                    for i, score in enumerate(scores):
                        bar = '█' * (score // 10)
                        print(f"  {i}: {score:3d} {bar}")
                else:
                    print("Failed to receive result from STM32")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def display_image(image):
    """Display image in console using ASCII art"""
    print("\nImage preview:")
    for row in range(0, 32, 2):
        line = ""
        for col in range(0, 32, 1):
            pixel = image[row, col] if row < 32 else 0
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


def batch_test(comm, num_samples=100):
    """Run batch testing"""
    print(f"\nRunning batch test with {num_samples} samples...")
    
    images, labels = load_test_images(num_samples)
    
    correct = 0
    total_time = 0
    
    for i, (image, true_label) in enumerate(zip(images, labels)):
        start_time = time.time()
        
        if comm.send_image(image):
            result = comm.receive_result()
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            if result:
                predicted, _ = result
                if predicted == true_label:
                    correct += 1
                
                status = "✓" if predicted == true_label else "✗"
                print(f"[{i+1}/{num_samples}] True: {true_label}, Pred: {predicted} {status} ({inference_time*1000:.0f}ms)")
            else:
                print(f"[{i+1}/{num_samples}] Failed to receive result")
        else:
            print(f"[{i+1}/{num_samples}] Failed to send image")
    
    accuracy = correct / num_samples * 100
    avg_time = total_time / num_samples * 1000
    
    print(f"\n{'='*50}")
    print(f"Results: {correct}/{num_samples} correct ({accuracy:.1f}% accuracy)")
    print(f"Average inference time: {avg_time:.1f}ms")


def main():
    """Main function"""
    print("=" * 60)
    print("EE 4065 - Homework 6: STM32 Serial Communication")
    print("Handwritten Digit Recognition Tester")
    print("=" * 60)
    
    # Create communication handler
    comm = STM32SerialComm()
    
    try:
        # List available ports
        ports = comm.list_ports()
        
        if not ports:
            print("\nNo COM ports found!")
            print("Make sure the STM32 Nucleo board is connected via USB.")
            return
        
        # Select port
        print("\nEnter COM port (or press Enter for auto-detect):")
        port_input = input("> ").strip()
        
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
            port = None  # Auto-detect
        
        # Connect
        comm.connect(port)
        
        # Wait for STM32 to be ready
        print("\nWaiting for STM32 to be ready...")
        print("(If this hangs, make sure the STM32 firmware is running)")
        
        # Main menu
        while True:
            print("\n" + "=" * 50)
            print("Options:")
            print("  1. Interactive test (single images)")
            print("  2. Batch test (100 random images)")
            print("  3. Quit")
            print("=" * 50)
            
            choice = input("Select option: ").strip()
            
            if choice == '1':
                interactive_test(comm)
            elif choice == '2':
                batch_test(comm)
            elif choice == '3':
                break
            else:
                print("Invalid option")
    
    except Exception as e:
        print(f"\nError: {e}")
    
    finally:
        comm.disconnect()


if __name__ == "__main__":
    main()

