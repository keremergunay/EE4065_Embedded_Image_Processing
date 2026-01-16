# Q1: Thresholding - Object Detection

## Overview

This project implements a thresholding-based object detection algorithm that finds bright objects with a target area of approximately **1000 pixels**. The algorithm searches through all possible threshold values (0-255) to find the one that yields an area closest to the target.

---

## Algorithm

```
1. Convert image to grayscale
2. For each threshold T from 255 to 0:
   a. Create binary image: pixel > T → white, else black
   b. Find largest connected component
   c. Calculate area (number of white pixels)
   d. If area is closest to 1000, save this threshold
3. Apply best threshold to create final binary mask
4. Keep only the largest connected component
5. Report: FOUND if area is within ±50 of 1000, else NOT FOUND
```

---

## Project Structure

```
Q1_Thresholding/
├── q1a_thresholding/           # Python Implementation
│   ├── q1a_thresholding.py
│   ├── resim.jpg               # Input image
│   ├── output_original.png     # Grayscale output
│   ├── output_binary.png       # Binary mask
│   └── output_overlay.png      # Green overlay on original
│
└── q1b_thresholding/           # ESP32-CAM Implementation
    └── q1b_thresholding.ino
```

---

## Q1a: Python Implementation

### Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Usage
```python
# Edit the image path in q1a_thresholding.py
img_path = r"path/to/your/image.jpg"

# Run
python q1a_thresholding.py
```

### Output
```
Image size: 176x127 (22352 pixels)
Target area: 1000 pixels (±50)

Threshold: 110
Detected area: 1003 pixels
Difference: 3 pixels
Status: FOUND
```

---

## Q1b: ESP32-CAM Implementation

### Features
- Real-time camera capture
- Web-based interface
- WiFi connectivity
- Morphological erosion for noise reduction
- Area adjustment (±50 pixels to reach exactly 1000)

### WiFi Configuration
```cpp
WiFi.begin("iPhone SE", "404404404");  // Station mode
```

### Web Interface
- **URL:** `http://<ESP32_IP>/`
- **Buttons:** Refresh, Find Object
- **Display:** Original image, Binary result, Statistics

### Key Functions

| Function | Description |
|----------|-------------|
| `searchThreshold()` | Tries all threshold values (255→0) to find best match |
| `findLargestComponent()` | Connected component labeling with flood fill |
| `keepLargestComponent()` | Removes small noise, keeps only main object |

---

## Technical Details

### Thresholding
```cpp
binary[i] = (gray[i] > T) ? 255 : 0;
```
- Pixels brighter than threshold T become white (foreground)
- Darker pixels become black (background)

### Connected Component Labeling
- 4-connectivity flood fill
- Stack-based implementation (avoids recursion overflow)
- Finds area of each connected region

### Area Adjustment
If detected area is within ±50 of target:
- **Too few pixels:** Add boundary pixels (dilation-like)
- **Too many pixels:** Remove boundary pixels (erosion-like)

---

## Why Tolerance?

The ±50 pixel tolerance is added to make the algorithm **generalizable** across different images and conditions:

1. **Real-world variations:** Objects rarely have exactly 1000 pixels due to lighting, camera angle, and resolution differences.

2. **Discrete threshold values:** Since thresholds are integers (0-255), finding an exact area match is often impossible. The closest threshold might yield 998 or 1003 pixels instead of exactly 1000.

3. **Boundary effects:** Connected component labeling can produce slightly different areas depending on edge pixels and anti-aliasing.

4. **Robustness:** A strict "exactly 1000 pixels" requirement would fail in most real-world scenarios. The tolerance allows the algorithm to work reliably across various input images.

---

## Why Erosion? (ESP32-CAM Only)

Morphological erosion is applied in the ESP32-CAM implementation to handle a common problem:

**Problem:** When thresholding, semi-bright areas around the object may also pass the threshold, creating unwanted connections between the object and background noise.

**Solution:** Erosion removes thin connections by requiring each pixel to have at least 2 white neighbors to survive:

```cpp
// A pixel stays white only if it has ≥2 white neighbors
if (whiteNeighbors >= 2) {
    eroded[idx] = 255;
}
```

**Effect:**
- Isolated pixels are removed
- Thin bridges between regions are broken
- Only solid, compact objects remain

This is especially important for real-time camera applications where lighting conditions vary.

---

## Results

| Parameter | Value |
|-----------|-------|
| Target Area | 1000 pixels |
| Tolerance | ±50 pixels |
| Threshold Range | 0-255 |
| Connectivity | 4 |

### Example Output
- **Input:** Grayscale image with bright object
- **Threshold Found:** 110
- **Detected Area:** 1003 pixels
- **Status:** FOUND ✓

---

## Important Code Details

### 1. Threshold Search Direction (High → Low)

```cpp
for (int t = 255; t >= 0; t--)
```

We search from **255 down to 0** because:
- The object is defined as **bright** (higher pixel values)
- Starting high ensures we find the tightest threshold that isolates just the object
- Going low would include more and more background pixels

### 2. Stack-Based Flood Fill

```cpp
Point ccStack[2000];  // Stack for flood fill
while (stackPtr > 0) {
    Point p = ccStack[--stackPtr];
    // ... process pixel ...
    ccStack[stackPtr++] = (Point){x+1, y};  // Add neighbors
}
```

**Why stack instead of recursion?**
- Recursion would cause **stack overflow** for large connected regions
- ESP32 has limited stack size (~8KB)
- Stack-based approach uses heap memory instead

### 3. 4-Connectivity vs 8-Connectivity

```cpp
// 4-connectivity: only cardinal directions
ccStack[stackPtr++] = (Point){x+1, y};  // Right
ccStack[stackPtr++] = (Point){x-1, y};  // Left
ccStack[stackPtr++] = (Point){x, y+1};  // Down
ccStack[stackPtr++] = (Point){x, y-1};  // Up
```

We use **4-connectivity** (not 8):
- More strict definition of "connected"
- Diagonal pixels are NOT considered neighbors
- Reduces false connections through corners

### 4. BMP Format for Web Display

```cpp
int createBMP(uint8_t *gray, int w, int h, uint8_t **outBmp)
```

**Why BMP instead of JPEG/PNG?**
- No compression needed = simpler code
- Works directly in browsers without JavaScript
- 8-bit grayscale palette = small file size
- No external libraries required

### 5. Camera Auto-Exposure Settings

```cpp
sensor_t *s = esp_camera_sensor_get();
s->set_exposure_ctrl(s, 1);  // Auto exposure ON
s->set_gain_ctrl(s, 1);      // Auto gain ON
s->set_aec2(s, 1);           // AEC DSP ON
```

These settings allow the camera to **automatically adapt** to different lighting conditions, ensuring consistent image quality.

### 6. Turkish Character Handling (Python)

```python
with open(img_path, 'rb') as f:
    img_array = np.frombuffer(f.read(), dtype=np.uint8)
gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
```

OpenCV's `imread()` fails with Turkish characters (ı, ş, ğ, etc.) in file paths. Solution:
- Read file with Python's native `open()`
- Decode with `cv2.imdecode()`

