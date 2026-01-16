"""
ResNet Model for MNIST Handwritten Digit Recognition
Adapted for STM32 Nucleo-F446RE (limited memory: 128KB RAM, 512KB Flash)
"""

import tensorflow as tf
from keras import layers, Model
from keras.regularizers import l2
from typing import Tuple


def resnet_layer(inputs, num_filters: int = 16, kernel_size: int = 3, 
                 strides: int = 1, activation: str = 'relu',
                 batch_normalization: bool = True, conv_first: bool = True):
    """
    2D Convolution-Batch Normalization-Activation stack builder
    
    Args:
        inputs: Input tensor
        num_filters: Conv2D number of filters
        kernel_size: Conv2D square kernel dimensions
        strides: Conv2D square stride dimensions
        activation: Activation name
        batch_normalization: Whether to include batch normalization
        conv_first: Conv-BN-Activation (True) or BN-Activation-Conv (False)
    
    Returns:
        Output tensor
    """
    conv = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(1e-4)
    )
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x


def ResNet(input_shape: Tuple[int, int, int] = (32, 32, 3),
           classes: int = 10, depth: int = 20, dropout: float = 0.2) -> Model:
    """
    ResNet v1 model adapted for MNIST
    
    Args:
        input_shape: Input image shape
        classes: Number of output classes
        depth: Network depth (should be 6n+2 for ResNet v1)
        dropout: Dropout rate
    
    Returns:
        Keras Model
    """
    if (depth - 2) % 6 != 0:
        raise ValueError("Depth should be 6n+2 (e.g., 8, 14, 20, 26, 32).")
    
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    # Stack of residual blocks
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            
            if stack > 0 and res_block == 0:
                # Linear projection for dimension matching
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False
                )
            
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
        
        num_filters *= 2
    
    # Classification head
    x = layers.AveragePooling2D(pool_size=4)(x)
    x = layers.Flatten()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(classes, activation='softmax',
                          kernel_initializer='he_normal',
                          name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f"ResNet{depth}")
    return model


def ResNetMini(input_shape: Tuple[int, int, int] = (32, 32, 3),
               classes: int = 10, dropout: float = 0.2) -> Model:
    """
    Minimal ResNet for very constrained MCUs like F446RE
    Depth = 8 (smallest possible ResNet)
    """
    return ResNet(input_shape=input_shape, classes=classes, depth=8, dropout=dropout)


def ResNet8(input_shape: Tuple[int, int, int] = (32, 32, 3),
            classes: int = 10, dropout: float = 0.2) -> Model:
    """ResNet with depth 8"""
    return ResNet(input_shape=input_shape, classes=classes, depth=8, dropout=dropout)


def ResNet14(input_shape: Tuple[int, int, int] = (32, 32, 3),
             classes: int = 10, dropout: float = 0.2) -> Model:
    """ResNet with depth 14"""
    return ResNet(input_shape=input_shape, classes=classes, depth=14, dropout=dropout)


def ResNet20(input_shape: Tuple[int, int, int] = (32, 32, 3),
             classes: int = 10, dropout: float = 0.2) -> Model:
    """ResNet with depth 20"""
    return ResNet(input_shape=input_shape, classes=classes, depth=20, dropout=dropout)

