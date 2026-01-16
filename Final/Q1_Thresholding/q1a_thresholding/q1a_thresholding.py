"""
Q1b - Simple Thresholding Algorithm
Target: Find bright object with approximately 1000 pixels
"""
import cv2
import numpy as np

# Parameters
TARGET_AREA = 1000
TOLERANCE = 50  # ±50 pixels

def find_largest_component(binary):
    """Find the largest connected component and return its mask and area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    
    if n <= 1:
        return np.zeros_like(binary), 0
    
    # Find largest component (skip background label 0)
    largest_label = 1
    largest_area = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
    
    # Create mask for largest component
    mask = np.zeros_like(binary)
    mask[labels == largest_label] = 255
    
    return mask, largest_area

def search_threshold(gray, target_area):
    """
    Search for the best threshold that yields an area closest to target_area.
    Returns: best_threshold, best_area, result_mask
    """
    best_t = 0
    best_area = 0
    closest_diff = 999999
    
    # Try all thresholds from 255 to 0
    for t in range(255, -1, -1):
        # Binary threshold: pixel > t -> white
        binary = np.zeros_like(gray)
        binary[gray > t] = 255
        
        # Find largest component
        _, area = find_largest_component(binary)
        
        diff = abs(area - target_area)
        if diff < closest_diff:
            closest_diff = diff
            best_t = t
            best_area = area
    
    # Generate final result with best threshold
    final_binary = np.zeros_like(gray)
    final_binary[gray > best_t] = 255
    
    # Keep only largest component
    result_mask, final_area = find_largest_component(final_binary)
    
    return best_t, final_area, result_mask

def main():
    # Input image path - CHANGE THIS
    img_path = r"C:\Users\Tarık\Desktop\EE4065 Final\Q1_Thresholding\q1a_thresholding\resim.jpg"
    
    # Read as grayscale (handle Turkish characters in path)
    with open(img_path, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    if gray is None:
        print(f"Error: Cannot read image '{img_path}'")
        return
    
    print(f"Image size: {gray.shape[1]}x{gray.shape[0]} ({gray.size} pixels)")
    print(f"Target area: {TARGET_AREA} pixels (±{TOLERANCE})")
    print()
    
    # Search for best threshold
    threshold, area, mask = search_threshold(gray, TARGET_AREA)
    
    # Check if object is found
    diff = abs(area - TARGET_AREA)
    found = diff <= TOLERANCE
    
    # Print results
    print(f"Threshold: {threshold}")
    print(f"Detected area: {area} pixels")
    print(f"Difference: {diff} pixels")
    print(f"Status: {'FOUND' if found else 'NOT FOUND'}")
    
    # Save outputs to q1a folder (handle Turkish characters)
    output_dir = r"C:\Users\Tarık\Desktop\EE4065 Final\Q1_Thresholding\q1a_thresholding"
    
    def save_image(path, img):
        _, encoded = cv2.imencode('.png', img)
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
    
    save_image(f"{output_dir}/output_original.png", gray)
    save_image(f"{output_dir}/output_binary.png", mask)
    
    # Create overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0, 255, 0)  # Green for detected object
    save_image(f"{output_dir}/output_overlay.png", overlay)
    
    print()
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
