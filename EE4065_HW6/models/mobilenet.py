"""
MobileNetV2 Model for MNIST Handwritten Digit Recognition
Adapted for STM32 Nucleo-F446RE (limited memory: 128KB RAM, 512KB Flash)
"""

import tensorflow as tf
from keras import layers, Model
from keras.applications import MobileNetV2 as KerasMobileNetV2


def make_divisible(v: int, divisor: int = 8, min_value: int = None) -> int:
    """Ensure the number is divisible by divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def inverted_residual_block(x, expansion: int, out_channels: int, 
                            stride: int, block_id: int):
    """Inverted Residual Block for MobileNetV2"""
    in_channels = x.shape[-1]
    prefix = f'block_{block_id}_'
    
    # Expansion
    if expansion != 1:
        expand_channels = in_channels * expansion
        x_expanded = layers.Conv2D(expand_channels, (1, 1), padding='same', 
                                   use_bias=False, name=prefix + 'expand')(x)
        x_expanded = layers.BatchNormalization(name=prefix + 'expand_BN')(x_expanded)
        x_expanded = layers.ReLU(6., name=prefix + 'expand_relu')(x_expanded)
    else:
        x_expanded = x
        expand_channels = in_channels
    
    # Depthwise
    x_dw = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same',
                                   use_bias=False, name=prefix + 'depthwise')(x_expanded)
    x_dw = layers.BatchNormalization(name=prefix + 'depthwise_BN')(x_dw)
    x_dw = layers.ReLU(6., name=prefix + 'depthwise_relu')(x_dw)
    
    # Project
    x_proj = layers.Conv2D(out_channels, (1, 1), padding='same',
                           use_bias=False, name=prefix + 'project')(x_dw)
    x_proj = layers.BatchNormalization(name=prefix + 'project_BN')(x_proj)
    
    # Residual connection
    if stride == 1 and in_channels == out_channels:
        return layers.Add(name=prefix + 'add')([x, x_proj])
    return x_proj


def MobileNetV2(input_shape=(32, 32, 3), classes=10, alpha=0.35, dropout=0.2):
    """
    MobileNetV2 model adapted for MNIST on STM32
    
    Args:
        input_shape: Input image shape
        classes: Number of output classes
        alpha: Width multiplier (0.35 is smallest for MCU)
        dropout: Dropout rate
    
    Returns:
        Keras Model
    """
    input_layer = layers.Input(shape=input_shape)
    
    # First conv
    first_filters = make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_filters, (3, 3), strides=(2, 2), padding='same',
                      use_bias=False, name='Conv1')(input_layer)
    x = layers.BatchNormalization(name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)
    
    # Inverted residual blocks
    # (expansion, out_channels, num_blocks, stride)
    block_configs = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    
    block_id = 0
    for expansion, out_ch, num_blocks, stride in block_configs:
        out_channels = make_divisible(out_ch * alpha, 8)
        for i in range(num_blocks):
            current_stride = stride if i == 0 else 1
            x = inverted_residual_block(x, expansion, out_channels, current_stride, block_id)
            block_id += 1
    
    # Last conv
    last_filters = make_divisible(1280 * alpha, 8)
    x = layers.Conv2D(last_filters, (1, 1), padding='same', use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    if dropout:
        x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(input_layer, x, name="MobileNetV2")
    return model


def MobileNetV2Mini(input_shape=(32, 32, 3), classes=10, dropout=0.2):
    """
    Minimal MobileNetV2 for very constrained MCUs like F446RE
    Fewer blocks and smaller width
    """
    input_layer = layers.Input(shape=input_shape)
    
    alpha = 0.25  # Very small width multiplier
    
    # First conv - smaller
    x = layers.Conv2D(8, (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    # Reduced blocks
    x = inverted_residual_block(x, 1, 8, 1, 0)
    x = inverted_residual_block(x, 6, 16, 2, 1)
    x = inverted_residual_block(x, 6, 16, 1, 2)
    x = inverted_residual_block(x, 6, 24, 2, 3)
    x = inverted_residual_block(x, 6, 24, 1, 4)
    x = inverted_residual_block(x, 6, 32, 2, 5)
    x = inverted_residual_block(x, 6, 32, 1, 6)
    
    # Last conv - smaller
    x = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(input_layer, x, name="MobileNetV2Mini")
    return model

