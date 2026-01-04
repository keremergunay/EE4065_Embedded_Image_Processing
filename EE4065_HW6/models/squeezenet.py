"""
SqueezeNet Model for MNIST Handwritten Digit Recognition
Adapted for STM32 Nucleo-F446RE (limited memory: 128KB RAM, 512KB Flash)
"""

import tensorflow as tf
from keras import layers, Model
from keras.utils import get_file

WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"


def fire_module(x, squeeze_filters, expand_filters, name):
    """Fire module: squeeze layer followed by expand layer"""
    # Squeeze layer
    squeeze = layers.Conv2D(
        filters=squeeze_filters,
        kernel_size=1,
        activation="relu",
        padding="same",
        name=f"{name}_squeeze"
    )(x)
    
    # Expand 1x1
    expand_1x1 = layers.Conv2D(
        filters=expand_filters,
        kernel_size=1,
        activation="relu",
        padding="same",
        name=f"{name}_expand1x1"
    )(squeeze)
    
    # Expand 3x3
    expand_3x3 = layers.Conv2D(
        filters=expand_filters,
        kernel_size=3,
        activation="relu",
        padding="same",
        name=f"{name}_expand3x3"
    )(squeeze)
    
    # Concatenate
    return layers.concatenate([expand_1x1, expand_3x3], name=f"{name}_concat")


def SqueezeNet(input_shape=(32, 32, 3), classes=10, weights=None, dropout=0.5):
    """
    SqueezeNet model adapted for MNIST classification
    
    Args:
        input_shape: Input image shape (default 32x32x3 for resized MNIST)
        classes: Number of output classes (10 for MNIST)
        weights: 'imagenet' for pretrained weights or None
        dropout: Dropout rate before final classification
    
    Returns:
        Keras Model
    """
    model_input = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=2, padding="valid", activation="relu", name="conv1")(model_input)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    # Fire modules
    x = fire_module(x, 16, 64, name="fire1")
    x = fire_module(x, 16, 64, name="fire2")
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    x = fire_module(x, 32, 128, name="fire3")
    x = fire_module(x, 32, 128, name="fire4")
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    x = fire_module(x, 48, 192, name="fire5")
    x = fire_module(x, 48, 192, name="fire6")
    x = fire_module(x, 64, 256, name="fire7")
    feature_extractor = fire_module(x, 64, 256, name="fire8")
    
    # Create feature extractor model for loading pretrained weights
    feature_ext_model = Model(inputs=model_input, outputs=feature_extractor)
    
    if weights == "imagenet":
        weights_path = get_file(
            "squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5",
            WEIGHTS_PATH_NO_TOP,
            cache_subdir="models",
        )
        feature_ext_model.load_weights(weights_path)
    
    # Classification head
    x = feature_ext_model.output
    if dropout:
        x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1), name="final_conv")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    model_output = layers.Softmax(name="predictions")(x)
    
    model = Model(inputs=model_input, outputs=model_output, name="SqueezeNet")
    
    return model


def SqueezeNetMini(input_shape=(32, 32, 3), classes=10, dropout=0.3):
    """
    Smaller SqueezeNet variant for very constrained MCUs like F446RE
    Reduced number of fire modules and filters
    """
    model_input = layers.Input(shape=input_shape)
    
    # Initial convolution (smaller)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv1")(model_input)
    x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
    
    # Reduced fire modules
    x = fire_module(x, 8, 32, name="fire1")
    x = fire_module(x, 8, 32, name="fire2")
    x = layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
    
    x = fire_module(x, 16, 64, name="fire3")
    x = fire_module(x, 16, 64, name="fire4")
    
    # Classification head
    if dropout:
        x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1), name="final_conv")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    model_output = layers.Softmax(name="predictions")(x)
    
    model = Model(inputs=model_input, outputs=model_output, name="SqueezeNetMini")
    
    return model

