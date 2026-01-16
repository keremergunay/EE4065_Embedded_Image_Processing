# EE4065 – Embedded Digital Image Processing
## Homework 3
**Due Date:** December 19, 2025 — 23:59

**Team Members:** 
* Kerem Ergünay
* Tarık Erdoğan

---

## 📋 Description
This project implements **Otsu’s Thresholding Method** (for both grayscale and color images) and fundamental **Morphological Operations** (Erosion, Dilation, Opening, Closing) on an ARM Cortex-M microcontroller (STM32).

The system features a robust **UART communication protocol** allowing a PC-side Python script to:
1.  Trigger on-board processing for Grayscale Otsu.
2.  Transfer full-color images to STM32 RAM for Color Otsu processing and retrieve the results.
3.  Request morphological operations on the binary image formed in memory.

---

## 🛠️ Hardware & Software
* **Hardware:** STM32 Development Board (ARM Cortex-M)
* **Software (PC):**
    * Python 3.x (w/ PySerial, Matplotlib, Numpy, Pillow)
    * STM32CubeIDE
* **Communication:** UART (115200 baud) for command and image data transfer.

---

## Q1 — Otsu’s Thresholding (Grayscale) (20 pts)
### 🔹 Objective
Implement Otsu’s method to automatically determine the optimal threshold for a grayscale image stored in the microcontroller's flash memory and convert it to binary.

### 🔹 STM32 Implementation (`otsu_gray_process`)
The function `otsu_gray_process` in `main.c` performs the following steps:
1.  **Histogram Calculation:** Computes the histogram of the input grayscale image.
2.  **Optimal Thresholding:** Iterates through all possible threshold values (0-255) to find the one that maximizes the **inter-class variance** (between background and foreground).
3.  **Binarization:** Applies this threshold to create a binary image (`binary_image` buffer).

### 🔹 Python Logic (`otsu_verify.py`)
*   Sends command `'1'` to STM32.
*   Receives the processed binary image (160x120 bytes) via UART.
*   Displays the result using Matplotlib.

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/7f1621d2-055a-41a0-96c7-202ef5dc59eb" />


---

## Q2 — Otsu’s Thresholding (Color) (20 pts)
### 🔹 Objective
Extend Otsu’s method to color images by applying the thresholding logic independently to each RGB channel. This task demonstrates bidirectional image transfer between PC and STM32.

### 🔹 Communication Protocol
Since the color image is large (160x120x3 bytes), a specific handshake protocol is used:
1.  **Initiate:** Python sends command `'2'`. STM32 responds with `'A'` (Ack).
2.  **Upload:** Python sends the raw RGB byte stream. STM32 stores it in `rgb_image` buffer in RAM.
3.  **Process:** STM32 runs `process_color_otsu`.
    *   Calculates Otsu threshold for **Red**, **Green**, and **Blue** channels independently.
    *   Thresholds each channel in-place.
4.  **Complete:** STM32 sends `'D'` (Done) when finished.
5.  **Retrieve:** Python sends command `'3'` to request the result and displays the segmented color image.

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/416514b4-e196-4ad9-8e5d-776cba1c7c1d" />

---

## Q3 — Morphological Operations (60 pts)
### 🔹 Objective
Implement four fundamental morphological operations on the binary image generated in Q1. The operations uses a **3x3 Square Structuring Element (SE)**.

### 🔹 STM32 Implementation
All functions operate on the `binary_image` (Q1 result) and output to `morph_result`.
*   **Erosion (Command '4'):** Pixel is White (255) only if **ALL** pixels in the 3x3 neighborhood are White. Otherwise, it becomes Black (0).
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/46f62eb2-cced-40b1-bec4-9795e0cb6530" />

*   **Dilation (Command '5'):** Pixel is White (255) if **ANY** pixel in the 3x3 neighborhood is White.
    <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c1321cf1-b3e8-4108-bce3-35d2aec2bf77" />

  
*   **Opening (Command '6'):** Erosion followed by Dilation. Useful for removing small noise.
  <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/0963c5d4-2e0f-46e9-95e8-191bc6c7c183" />

*   **Closing (Command '7'):** Dilation followed by Erosion. Useful for filling small holes.
  <img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/d208ee10-8c46-4a67-94b7-392b92aa5fbf" />


### 🔹 Verification
The Python script provides a menu to select these operations. It first ensures Q1 has been run (to populate the binary image in STM32 memory), then sends the corresponding command code ('4', '5', '6', or '7') and visualizes the returned result.


---

## 🧪 Testing & User Interface
The project includes a comprehensive Python CLI tool (`otsu_verify.py`) to interact with the board.

### 🔹 Menu System
```text
========================================
EE4065 - HW3 Menu
1. Run Question 1 (Grayscale Otsu)
2. Run Question 2 (Color Otsu)
3. Run Question 3 (Morphological Ops)
q. Exit
========================================
```

### 🔹 Example Workflows
1.  **Grayscale Otsu:** User calculates the binary version of the hardcoded `mandrill` image.
2.  **Color Otsu:** User verifies the system can handle dynamic image data transfer and multi-channel processing.
3.  **Morphology:** User observes the effects of erosion/dilation on the binary features extracted in Q1.





