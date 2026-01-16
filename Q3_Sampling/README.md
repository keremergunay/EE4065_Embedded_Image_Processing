# Q3: Image Resampling with Bilinear Interpolation

## Overview

This project implements **upsampling** and **downsampling** operations on the ESP32-CAM using bilinear interpolation. The algorithm supports **non-integer scale factors** such as 1.5 and 2/3.

---

## Algorithm

### Bilinear Interpolation

For each output pixel `(x_out, y_out)`:

```
1. Map to source coordinates:
   x_in = x_out / scale
   y_in = y_out / scale

2. Find 4 neighboring pixels:
   (x0, y0), (x1, y0), (x0, y1), (x1, y1)
   where x0 = floor(x_in), x1 = x0+1

3. Calculate weights:
   dx = x_in - x0
   dy = y_in - y0

4. Interpolate:
   p = (1-dx)(1-dy)·I00 + dx(1-dy)·I10 + (1-dx)dy·I01 + dx·dy·I11
```

### Why Same Formula for Up/Down?

The bilinear formula works for **both upsampling and downsampling**:

| Operation | Scale | Effect |
|-----------|-------|--------|
| **Upsample** | > 1 | Multiple output pixels map to same source region |
| **Downsample** | < 1 | Output pixels skip source pixels |

The mapping `x_in = x_out / scale` automatically handles both directions.

---

## API

Two APIs are provided:

```cpp
// API A: Allocate buffer (caller must free)
uint8_t* resample_bilinear_alloc(
    const uint8_t *src, int src_w, int src_h,
    float scale, int *dst_w, int *dst_h
);

// API B: Write to preallocated buffer
bool resample_bilinear_into(
    const uint8_t *src, int src_w, int src_h,
    float scale, uint8_t *dst, int *dst_w, int *dst_h
);
```

---

## Usage Examples

### Upsampling (scale = 1.5)
```cpp
int out_w, out_h;
resample_bilinear_into(img, 160, 120, 1.5, output, &out_w, &out_h);
// Result: 240x180 pixels
```

### Downsampling (scale = 2/3)
```cpp
int out_w, out_h;
resample_bilinear_into(img, 160, 120, 0.667, output, &out_w, &out_h);
// Result: 107x80 pixels
```

---

## ESP32-CAM Implementation

### Features
- Real-time camera capture (QQVGA 160x120)
- Web-based interface with horizontal layout
- WiFi connectivity
- BMP image display

### Scale Options
| Button | Scale | Input | Output | Bytes |
|--------|-------|-------|--------|-------|
| x1.5 | 1.5 | 160x120 | 240x180 | 43,200 |
| x0.5 | 0.5 | 160x120 | 80x60 | 4,800 |
| x2/3 | 0.667 | 160x120 | 107x80 | 8,560 |

### Memory
- Output buffer: 45,000 bytes
- Maximum safe upsample: x1.5

---

## Technical Details

### Border Handling
Coordinates outside image bounds are **clamped** to valid range:
```cpp
x0 = constrain(x0, 0, src_w - 1);
y0 = constrain(y0, 0, src_h - 1);
x1 = constrain(x1, 0, src_w - 1);
y1 = constrain(y1, 0, src_h - 1);
```

### Output Size Calculation
```cpp
dst_w = round(src_w * scale);
dst_h = round(src_h * scale);
```

### Pixel Value
Final value is rounded and clamped to [0, 255]:
```cpp
int val = (int)round(p);
val = constrain(val, 0, 255);
```

---

## File Structure

```
Q3_Sampling/
└── q3_sampling_esp32/
    └── q3_sampling_esp32.ino    # ESP32-CAM implementation
```

---

## Results

| Parameter | Value |
|-----------|-------|
| Input Resolution | 160x120 |
| Supported Scales | 0.25 - 1.5 |
| Interpolation | Bilinear |
| Format | 8-bit Grayscale |
