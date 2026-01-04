"""
EfficientNet Model for MNIST Handwritten Digit Recognition
Adapted for STM32 Nucleo-F446RE (limited memory: 128KB RAM, 512KB Flash)
Based on STMicroelectronics implementation
"""

import tensorflow as tf
from keras import layers, Model
from typing import List, Tuple
import math


def round_filters(filters: int, width_coefficient: float, depth_divisor: int = 8) -> int:
    """Round number of filters based on width coefficient."""
    if not width_coefficient:
        return filters
    filters *= width_coefficient
    new_filters = max(depth_divisor, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats: List[int]) -> List[int]:
    """Scale number of repeats."""
    num_repeat = sum(repeats)
    num_repeat_scaled = int(math.ceil(num_repeat))
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    return repeats_scaled[::-1]


def mb_conv_block(inputs: tf.Tensor, in_channels: int, out_channels: int, 
                  num_repeat: int, stride: int, expansion_factor: int, 
                  se_ratio: float, kernel_size: int, drop_rate: float,
                  prev_block_num: int, activation) -> tf.Tensor:
    """
    Mobile Inverted Bottleneck Convolution Block
    """
    x = inputs
    input_filters = in_channels
    
    for i in range(num_repeat):
        input_tensor = x
        current_stride = stride if i == 0 else 1
        
        expanded_filters = input_filters * expansion_factor
        
        # Expansion phase
        if expansion_factor != 1:
            x = layers.Conv2D(expanded_filters, (1, 1), strides=1, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
        
        # Depthwise convolution
        x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=current_stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        # Squeeze and Excitation
        squeezed_filters = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, expanded_filters))(se)
        se = layers.Conv2D(squeezed_filters, (1, 1), padding='same', activation=activation)(se)
        se = layers.Conv2D(expanded_filters, (1, 1), padding='same')(se)
        # Clip values for quantization compatibility
        se = layers.Activation('sigmoid')(se)
        x = layers.multiply([x, se])
        
        # Output phase
        x = layers.Conv2D(out_channels, (1, 1), strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection
        if current_stride == 1 and input_filters == out_channels:
            if drop_rate and drop_rate > 0:
                x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
            x = layers.add([x, input_tensor])
        
        input_filters = out_channels
    
    return x


def EfficientNet(input_shape: Tuple[int, int, int] = (32, 32, 3),
                 classes: int = 10,
                 width_coefficient: float = 0.45,
                 dropout_rate: float = 0.2,
                 se_ratio: float = 0.25,
                 drop_connect_rate: float = 0.2) -> Model:
    """
    EfficientNet model adapted for MNIST on STM32
    
    Args:
        input_shape: Input image shape
        classes: Number of output classes
        width_coefficient: Width scaling factor (reduced for MCU)
        dropout_rate: Dropout rate
        se_ratio: Squeeze-excitation ratio
        drop_connect_rate: Drop connect rate
    
    Returns:
        Keras Model
    """
    repeats = [1, 2, 2, 3, 3, 4, 1]
    input_layer = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(round_filters(32, width_coefficient), (3, 3), strides=(2, 2), 
                      padding='same', use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.relu6, name='stem_activation')(x)
    
    # Build blocks
    repeats_scaled = round_repeats(repeats)
    exp_ratios = [1] + [3] * (len(repeats_scaled) - 1)
    
    # Block configurations: (in_ch, out_ch, stride, kernel)
    block_configs = [
        (32, 16, 1, 3),
        (16, 24, 2, 3),
        (24, 40, 2, 5),
        (40, 80, 2, 3),
        (80, 112, 1, 5),
        (112, 192, 2, 5),
        (192, 320, 1, 3),
    ]
    
    prev_block_num = 0
    for i, (in_ch, out_ch, stride, kernel) in enumerate(block_configs):
        x = mb_conv_block(
            x,
            in_channels=round_filters(in_ch, width_coefficient),
            out_channels=round_filters(out_ch, width_coefficient),
            num_repeat=repeats_scaled[i],
            stride=stride,
            expansion_factor=exp_ratios[i],
            se_ratio=se_ratio,
            kernel_size=kernel,
            drop_rate=drop_connect_rate,
            prev_block_num=prev_block_num,
            activation=tf.nn.relu6
        )
        prev_block_num += repeats_scaled[i]
    
    # Top
    x = layers.Conv2D(round_filters(1280, width_coefficient), (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.relu6, name='top_activation')(x)
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(input_layer, x, name="EfficientNet")
    return model


def EfficientNetMini(input_shape: Tuple[int, int, int] = (32, 32, 3),
                     classes: int = 10,
                     dropout_rate: float = 0.2) -> Model:
    """
    Minimal EfficientNet for very constrained MCUs like F446RE
    """
    input_layer = layers.Input(shape=input_shape)
    
    # Stem - reduced
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.relu6)(x)
    
    # Simplified blocks
    x = mb_conv_block(x, 16, 16, 1, 1, 1, 0.25, 3, 0.1, 0, tf.nn.relu6)
    x = mb_conv_block(x, 16, 24, 2, 2, 3, 0.25, 3, 0.1, 1, tf.nn.relu6)
    x = mb_conv_block(x, 24, 32, 2, 2, 3, 0.25, 3, 0.1, 3, tf.nn.relu6)
    x = mb_conv_block(x, 32, 48, 1, 2, 3, 0.25, 3, 0.1, 5, tf.nn.relu6)
    
    # Top - reduced
    x = layers.Conv2D(96, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.relu6)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(input_layer, x, name="EfficientNetMini")
    return model

