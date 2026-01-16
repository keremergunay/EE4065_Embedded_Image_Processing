"""
Question 5b BONUS - SSD + MobileNet for Digit Detection

SSD (Single Shot MultiBox Detector) combined with MobileNet backbone.
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


class SSDMobileNet:
    """
    SSD with MobileNet backbone for digit detection.
    
    Architecture:
    - MobileNet backbone for feature extraction
    - Multi-scale feature maps
    - Anchor-based detection heads
    """
    
    def __init__(self, input_shape=(160, 160, 1), num_classes=10, num_anchors=4):
        """
        Initialize SSD-MobileNet model.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of digit classes (0-9)
            num_anchors: Number of anchor boxes per location
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.model = None
        
        if HAS_TF:
            self.model = self._build_model()
    
    def _mobilenet_block(self, x, filters, stride=1):
        """MobileNet depthwise separable conv block."""
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x
    
    def _detection_head(self, feature_map, name_prefix):
        """
        Detection head for SSD.
        Outputs class probabilities and bbox offsets.
        """
        # Class prediction: (batch, H, W, num_anchors * (num_classes + 1))
        class_pred = layers.Conv2D(
            self.num_anchors * (self.num_classes + 1), 
            3, padding='same',
            name=f'{name_prefix}_class'
        )(feature_map)
        
        # Bbox prediction: (batch, H, W, num_anchors * 4) - [dx, dy, dw, dh]
        bbox_pred = layers.Conv2D(
            self.num_anchors * 4, 
            3, padding='same',
            name=f'{name_prefix}_bbox'
        )(feature_map)
        
        return class_pred, bbox_pred
    
    def _build_model(self):
        """Build SSD-MobileNet model."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv
        x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        
        # MobileNet backbone with multiple feature maps
        x = self._mobilenet_block(x, 64, stride=1)
        x = self._mobilenet_block(x, 64, stride=2)
        feat1 = x  # 40x40 (or similar based on input)
        
        x = self._mobilenet_block(x, 128, stride=1)
        x = self._mobilenet_block(x, 128, stride=2)
        feat2 = x  # 20x20
        
        x = self._mobilenet_block(x, 256, stride=1)
        x = self._mobilenet_block(x, 256, stride=2)
        feat3 = x  # 10x10
        
        x = self._mobilenet_block(x, 512, stride=1)
        x = self._mobilenet_block(x, 512, stride=2)
        feat4 = x  # 5x5
        
        # Detection heads at multiple scales
        class1, bbox1 = self._detection_head(feat1, 'det1')
        class2, bbox2 = self._detection_head(feat2, 'det2')
        class3, bbox3 = self._detection_head(feat3, 'det3')
        class4, bbox4 = self._detection_head(feat4, 'det4')
        
        # Reshape and concatenate all detections
        def reshape_predictions(class_pred, bbox_pred):
            batch_size = tf.shape(class_pred)[0]
            h, w = class_pred.shape[1], class_pred.shape[2]
            
            class_pred = tf.reshape(class_pred, [batch_size, -1, self.num_classes + 1])
            bbox_pred = tf.reshape(bbox_pred, [batch_size, -1, 4])
            
            return class_pred, bbox_pred
        
        class1, bbox1 = reshape_predictions(class1, bbox1)
        class2, bbox2 = reshape_predictions(class2, bbox2)
        class3, bbox3 = reshape_predictions(class3, bbox3)
        class4, bbox4 = reshape_predictions(class4, bbox4)
        
        # Concatenate
        all_classes = layers.Concatenate(axis=1)([class1, class2, class3, class4])
        all_bboxes = layers.Concatenate(axis=1)([bbox1, bbox2, bbox3, bbox4])
        
        # Apply softmax to class predictions
        all_classes = layers.Softmax()(all_classes)
        
        model = keras.Model(inputs, [all_classes, all_bboxes], name='ssd_mobilenet_digit')
        return model
    
    def generate_anchors(self):
        """
        Generate anchor boxes for all feature maps.
        
        Returns:
            anchors: (N, 4) array of [cx, cy, w, h] in normalized coords
        """
        feature_map_sizes = [40, 20, 10, 5]  # Approximate sizes
        anchor_scales = [0.1, 0.2, 0.4, 0.6]
        anchor_ratios = [1.0, 0.5, 2.0, 1.0]  # num_anchors ratios
        
        anchors = []
        
        for fm_idx, fm_size in enumerate(feature_map_sizes):
            scale = anchor_scales[fm_idx]
            
            for i in range(fm_size):
                for j in range(fm_size):
                    cx = (j + 0.5) / fm_size
                    cy = (i + 0.5) / fm_size
                    
                    for ratio in anchor_ratios:
                        w = scale * np.sqrt(ratio)
                        h = scale / np.sqrt(ratio)
                        anchors.append([cx, cy, w, h])
        
        return np.array(anchors, dtype='float32')
    
    def decode_predictions(self, class_preds, bbox_preds, anchors, 
                          conf_threshold=0.5, nms_threshold=0.5):
        """
        Decode raw predictions to detections.
        
        Args:
            class_preds: (N, num_classes+1) class probabilities
            bbox_preds: (N, 4) bbox offsets [dx, dy, dw, dh]
            anchors: (N, 4) anchor boxes [cx, cy, w, h]
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
        
        Returns:
            List of (x1, y1, x2, y2, class_id, confidence)
        """
        detections = []
        
        for i in range(len(class_preds)):
            class_id = np.argmax(class_preds[i])
            conf = class_preds[i][class_id]
            
            # Skip background (last class) and low confidence
            if class_id == self.num_classes or conf < conf_threshold:
                continue
            
            # Decode bbox
            anchor = anchors[i]
            dx, dy, dw, dh = bbox_preds[i]
            
            cx = anchor[0] + dx * anchor[2]
            cy = anchor[1] + dy * anchor[3]
            w = anchor[2] * np.exp(dw)
            h = anchor[3] * np.exp(dh)
            
            # Convert to corner format
            x1 = max(0, cx - w/2)
            y1 = max(0, cy - h/2)
            x2 = min(1, cx + w/2)
            y2 = min(1, cy + h/2)
            
            detections.append([x1, y1, x2, y2, class_id, float(conf)])
        
        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections, nms_threshold)
        
        return detections
    
    def _nms(self, detections, threshold):
        """Non-maximum suppression."""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x[5], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [d for d in detections 
                         if self._iou(best[:4], d[:4]) < threshold]
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def detect(self, image, conf_threshold=0.5, nms_threshold=0.5):
        """
        Detect digits in image.
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold
        
        Returns:
            List of (x1, y1, x2, y2, class_id, confidence) in pixel coords
        """
        if self.model is None:
            return []
        
        # Preprocess
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        resized = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        normalized = resized.astype('float32') / 255.0
        if len(normalized.shape) == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        batch = np.expand_dims(normalized, axis=0)
        
        # Predict
        class_preds, bbox_preds = self.model.predict(batch, verbose=0)
        class_preds = class_preds[0]
        bbox_preds = bbox_preds[0]
        
        # Get anchors
        anchors = self.generate_anchors()
        
        # Decode
        detections = self.decode_predictions(
            class_preds, bbox_preds, anchors,
            conf_threshold, nms_threshold
        )
        
        # Scale to original image size
        scaled_detections = []
        for (x1, y1, x2, y2, cls, conf) in detections:
            scaled_detections.append((
                int(x1 * w), int(y1 * h),
                int(x2 * w), int(y2 * h),
                cls, conf
            ))
        
        return scaled_detections
    
    def draw_detections(self, image, detections, class_names=None):
        """Draw detection boxes on image."""
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for (x1, y1, x2, y2, cls, conf) in detections:
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if class_names and cls < len(class_names):
                label = f"{class_names[cls]}: {conf:.2f}"
            else:
                label = f"{cls}: {conf:.2f}"
            
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
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


def main():
    """Test SSD-MobileNet on digits."""
    if not HAS_TF:
        print("TensorFlow required")
        return
    
    print("Creating SSD-MobileNet model...")
    ssd = SSDMobileNet(input_shape=(160, 160, 1), num_classes=10)
    
    print(f"Model summary:")
    ssd.model.summary()
    
    print(f"\nGenerated {len(ssd.generate_anchors())} anchor boxes")
    
    # Test on sample image
    if HAS_CV2:
        # Create test image with drawn digit
        test_image = np.ones((160, 160), dtype=np.uint8) * 255
        cv2.putText(test_image, "5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        cv2.imwrite("q5b_test_input.png", test_image)
        
        print("\nNote: Model requires training for accurate detection.")
        print("This is a demonstration of the architecture.")
        
        # Would run detection after training:
        # detections = ssd.detect(test_image, conf_threshold=0.3)
        # result = ssd.draw_detections(test_image, detections)
        # cv2.imwrite("q5b_ssd_result.png", result)
        
        print("Saved test input to q5b_test_input.png")


if __name__ == "__main__":
    main()
