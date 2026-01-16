import numpy as np
import re

IMG_W = 160
IMG_H = 120
IMG_SIZE = IMG_W * IMG_H

def load_header_image(path):
    with open(path, "r") as f:
        txt = f.read()
    m = re.search(r"\{([^}]*)\}", txt, re.S)
    if not m:
        raise ValueError("Header içinde { ... } kısmı bulunamadı.")
    body = m.group(1)
    nums = re.findall(r"0x[0-9A-Fa-f]+|\d+", body)
    data = [int(x, 16) if x.lower().startswith("0x") else int(x) for x in nums]
    arr = np.array(data, dtype=np.uint8)
    if arr.size != IMG_SIZE:
        raise ValueError(f"Boyut uyuşmuyor: {arr.size} != {IMG_SIZE}")
    return arr

def calc_histogram(img):
    hist = np.bincount(img.astype(np.int32), minlength=256).astype(np.uint32)
    return hist

def hist_equalize(inImg):
    size = inImg.size
    hist_in = calc_histogram(inImg)
    cdf = np.cumsum(hist_in, dtype=np.uint32)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        num = cdf[i] * 255 + (size // 2)
        lut[i] = (num // size).astype(np.uint8)
    out = lut[inImg]
    hist_out = calc_histogram(out)
    return out, hist_in, hist_out

def conv3x3(inImg, kernel, divisor):
    w, h = IMG_W, IMG_H
    inp = inImg.reshape((h, w)).astype(np.int32)
    out = np.zeros_like(inp, dtype=np.int32)

    # kenarları input’tan kopyala
    out[0, :] = inp[0, :]
    out[h-1, :] = inp[h-1, :]
    out[:, 0] = inp[:, 0]
    out[:, w-1] = inp[:, w-1]

    for y in range(1, h-1):
        for x in range(1, w-1):
            s = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    s += inp[y+ky, x+kx] * kernel[ky+1][kx+1]
            if divisor != 0:
                s //= divisor
            if s < 0:
                s = 0
            if s > 255:
                s = 255
            out[y, x] = s

    return out.astype(np.uint8).reshape(-1)

def low_pass_filter(inImg):
    k = np.array([[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]], dtype=np.int32)
    return conv3x3(inImg, k, 9)

def high_pass_filter(inImg):
    k = np.array([[ 0, -1,  0],
                  [-1,  4, -1],
                  [ 0, -1,  0]], dtype=np.int32)
    return conv3x3(inImg, k, 1)

def median_filter3x3(inImg):
    w, h = IMG_W, IMG_H
    inp = inImg.reshape((h, w))
    out = np.zeros_like(inp, dtype=np.uint8)

    out[0, :] = inp[0, :]
    out[h-1, :] = inp[h-1, :]
    out[:, 0] = inp[:, 0]
    out[:, w-1] = inp[:, w-1]

    win = np.zeros(9, dtype=np.uint8)
    for y in range(1, h-1):
        for x in range(1, w-1):
            k = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    win[k] = inp[y+ky, x+kx]
                    k += 1
            win_sorted = np.sort(win)
            out[y, x] = win_sorted[4]

    return out.reshape(-1)

if __name__ == "__main__":
    # 1) header’dan resmi al
    img_in = load_header_image("C:\Users\Tarık\STM32CubeIDE\workspace2\HW2\Core\Inc\mandrill.h")  # gerekirse dosya adını değiştir

    # 2) histogram + equalization
    img_eq, hist_orig, hist_eq = hist_equalize(img_in)

    # 3) low / high / median filtreler
    img_low  = low_pass_filter(img_in)
    img_high = high_pass_filter(img_in)
    img_med  = median_filter3x3(img_in)

    # 4) Örnek karşılaştırma için birkaç index
    idxs = [0, 123, 5000, 10000, 19199]
    print("Index | in  eq  low  high  med")
    for i in idxs:
        print(f"{i:5d} | {img_in[i]:3d} {img_eq[i]:3d} {img_low[i]:3d} {img_high[i]:4d} {img_med[i]:4d}")

    # Histogramdan birkaç örnek
    print("\nHistogram örnekleri (orig / eq):")
    for g in [0, 10, 50, 100, 150, 200, 250]:
        print(f"g={g:3d}: orig={hist_orig[g]}, eq={hist_eq[g]}")
