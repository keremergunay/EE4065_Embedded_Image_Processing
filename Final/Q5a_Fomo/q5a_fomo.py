"""
Question 5a BONUS - FOMO (Faster Objects, More Objects) with Keras
Based on Edge Impulse FOMO architecture for object detection.

Reference: https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo
"""

import numpy as np
import os

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow required. Install with: pip install tensorflow")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FOMO:
    """
    FOMO (Faster Objects, More Objects) detector.
    
    Key features:
    - Fully convolutional architecture
    - Outputs heatmap instead of bounding boxes
    - Centroid-based object detection
    - Very efficient for embedded systems
    """
    
    def __init__(self, input_shape=(96, 96, 1), num_classes=10, grid_size=12):
        """
        Initialize FOMO model.
        
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of object classes
            grid_size: Output grid size (input_size / 8 typically)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.model = None
        
        if HAS_TF:
            self.model = self._build_model()
    
    def _build_model(self):
        """Build FOMO model architecture."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Backbone (MobileNet-style)
        x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Block 1
        x = self._depthwise_sep_block(x, 32, stride=2)
        x = self._depthwise_sep_block(x, 32, stride=1)
        
        # Block 2
        x = self._depthwise_sep_block(x, 64, stride=2)
        x = self._depthwise_sep_block(x, 64, stride=1)
        
        # Block 3 (no more downsampling, keep spatial info)
        x = self._depthwise_sep_block(x, 128, stride=1)
        x = self._depthwise_sep_block(x, 128, stride=1)
        
        # Detection head - outputs (grid_h, grid_w, num_classes + 1)
        # +1 for background/no-object class
        x = layers.Conv2D(self.num_classes + 1, 1, activation='softmax')(x)
        
        model = keras.Model(inputs, x, name='fomo_digit')
        return model
    
    def _depthwise_sep_block(self, x, filters, stride=1):
        """Depthwise separable convolution block."""
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=20):
        """
        Train FOMO model.
        
        Args:
            x_train: Training images (N, H, W, C)
            y_train: Training labels - heatmaps (N, grid_h, grid_w, num_classes+1)
            x_val: Validation images
            y_val: Validation labels
            epochs: Training epochs
        """
        if self.model is None:
            raise RuntimeError("TensorFlow not available")
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        validation_data = (x_val, y_val) if x_val is not None else None
        
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=validation_data,
            verbose=1
        )
    
    def detect(self, image, conf_threshold=0.5):
        """
        Detect objects in image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            conf_threshold: Confidence threshold
        
        Returns:
            List of (center_x, center_y, class_id, confidence)
        """
        if self.model is None:
            return []
        
        # Preprocess
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        normalized = resized.astype('float32') / 255.0
        if len(normalized.shape) == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        batch = np.expand_dims(normalized, axis=0)
        
        # Predict
        heatmap = self.model.predict(batch, verbose=0)[0]  # (grid_h, grid_w, num_classes+1)
        
        # Parse detections
        detections = []
        grid_h, grid_w = heatmap.shape[:2]
        
        for i in range(grid_h):
            for j in range(grid_w):
                probs = heatmap[i, j]
                class_id = np.argmax(probs)
                conf = probs[class_id]
                
                # Skip background class (last class) and low confidence
                if class_id < self.num_classes and conf >= conf_threshold:
                    # Convert grid coords to image coords
                    center_x = (j + 0.5) * self.input_shape[1] / grid_w
                    center_y = (i + 0.5) * self.input_shape[0] / grid_h
                    
                    detections.append((center_x, center_y, class_id, float(conf)))
        
        return detections
    
    def draw_detections(self, image, detections, class_names=None):
        """Draw detection centroids on image."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for (cx, cy, cls, conf) in detections:
            # Draw circle at centroid
            cv2.circle(result, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            
            # Label
            if class_names and cls < len(class_names):
                label = f"{class_names[cls]}: {conf:.2f}"
            else:
                label = f"{cls}: {conf:.2f}"
            
            cv2.putText(result, label, (int(cx) + 7, int(cy) - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return result
    
    def save(self, path):
        """Save model weights."""
        if self.model:
            self.model.save_weights(path)
    
    def load(self, path):
        """Load model weights."""
        if self.model and os.path.exists(path):
            self.model.load_weights(path)
            return True
        return False


def create_fomo_training_data(images, labels, grid_size=12, input_size=96):
    """
    Create FOMO training data from MNIST-style images.
    
    For simplicity, we place the digit at a random location
    and create corresponding heatmap labels.
    
    Args:
        images: MNIST images (N, 28, 28)
        labels: MNIST labels (N,)
        grid_size: Output grid size
        input_size: Input image size
    
    Returns:
        x_train: Images (N, input_size, input_size, 1)
        y_train: Heatmaps (N, grid_size, grid_size, 11)  # 10 digits + background
    """
    n_samples = len(images)
    x_train = np.zeros((n_samples, input_size, input_size, 1), dtype='float32')
    y_train = np.zeros((n_samples, grid_size, grid_size, 11), dtype='float32')
    
    for i in range(n_samples):
        # Random position for digit
        max_offset = input_size - 28
        offset_x = np.random.randint(0, max_offset + 1)
        offset_y = np.random.randint(0, max_offset + 1)
        
        # Place digit
        x_train[i, offset_y:offset_y+28, offset_x:offset_x+28, 0] = images[i] / 255.0
        
        # Create heatmap label
        center_x = offset_x + 14
        center_y = offset_y + 14
        
        grid_x = int(center_x * grid_size / input_size)
        grid_y = int(center_y * grid_size / input_size)
        
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)
        
        # Set class label at grid location
        y_train[i, grid_y, grid_x, labels[i]] = 1.0
        
        # Set background for all other locations
        for gy in range(grid_size):
            for gx in range(grid_size):
                if y_train[i, gy, gx].sum() == 0:
                    y_train[i, gy, gx, 10] = 1.0  # Background class
    
    return x_train, y_train


def main():
    """Train and test FOMO on digits."""
    if not HAS_TF:
        print("TensorFlow required")
        return
    
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Create FOMO training data
    print("Creating FOMO training data...")
    x_fomo_train, y_fomo_train = create_fomo_training_data(x_train[:5000], y_train[:5000])
    x_fomo_test, y_fomo_test = create_fomo_training_data(x_test[:500], y_test[:500])
    
    print(f"Training data: {x_fomo_train.shape}, {y_fomo_train.shape}")
    
    # Create and train FOMO
    fomo = FOMO(input_shape=(96, 96, 1), num_classes=10)
    
    if fomo.load('fomo_digit.h5'):
        print("Loaded pre-trained FOMO model")
    else:
        print("Training FOMO...")
        fomo.train(x_fomo_train, y_fomo_train, 
                  x_val=x_fomo_test, y_val=y_fomo_test, 
                  epochs=10)
        fomo.save('fomo_digit.h5')
    
    # Test detection
    print("\nTesting detection...")
    test_idx = 0
    test_image = x_fomo_test[test_idx, :, :, 0]
    detections = fomo.detect(test_image, conf_threshold=0.3)
    
    print(f"Found {len(detections)} detection(s)")
    for (cx, cy, cls, conf) in detections:
        print(f"  Digit {cls} at ({cx:.1f}, {cy:.1f}), confidence: {conf:.3f}")
    
    # Save visualization
    if HAS_CV2:
        vis = fomo.draw_detections((test_image * 255).astype(np.uint8), detections)
        cv2.imwrite('q5a_fomo_result.png', vis)
        print("Saved result to q5a_fomo_result.png")


if __name__ == "__main__":
    main()
