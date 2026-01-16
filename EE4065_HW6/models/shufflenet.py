"""
ShuffleNet Model for MNIST Handwritten Digit Recognition
Adapted for STM32 Nucleo-F446RE (limited memory: 128KB RAM, 512KB Flash)
"""

import tensorflow as tf
from keras import layers, Model
import numpy as np


def channel_shuffle(x, groups):
    """Channel shuffle operation for ShuffleNet"""
    batch_size, height, width, channels = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    channels_per_group = channels // groups
    
    # Reshape to (batch, h, w, groups, channels_per_group)
    x = tf.reshape(x, [batch_size, height, width, groups, channels_per_group])
    # Transpose to (batch, h, w, channels_per_group, groups)
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    # Reshape back to (batch, h, w, channels)
    x = tf.reshape(x, [batch_size, height, width, channels])
    return x


def group_conv(x, out_channels, groups=1, kernel=1, stride=1, name=""):
    """Group convolution layer"""
    if groups == 1:
        return layers.Conv2D(
            out_channels, kernel, strides=stride, padding='same',
            use_bias=False, name=name
        )(x)
    
    in_channels = x.shape[-1]
    channels_per_group = in_channels // groups
    out_per_group = out_channels // groups
    
    group_outputs = []
    for g in range(groups):
        start_ch = g * channels_per_group
        end_ch = start_ch + channels_per_group
        group_input = x[:, :, :, start_ch:end_ch]
        group_output = layers.Conv2D(
            out_per_group, kernel, strides=stride, padding='same',
            use_bias=False, name=f"{name}_g{g}"
        )(group_input)
        group_outputs.append(group_output)
    
    return layers.Concatenate(name=f"{name}_concat")(group_outputs)


def shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio,
                 strides=1, stage=1, block=1):
    """ShuffleNet unit"""
    prefix = f"stage{stage}_block{block}"
    
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    
    # For first block in stage (not stage 2), use groups=1
    first_groups = 1 if (stage == 2 and block == 1) else groups
    
    # Group conv 1x1
    x = layers.Conv2D(bottleneck_channels, (1, 1), padding='same', 
                      use_bias=False, name=f"{prefix}_gconv1")(inputs)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation('relu', name=f"{prefix}_relu1")(x)
    
    # Channel shuffle
    if groups > 1:
        x = layers.Lambda(lambda z: channel_shuffle(z, groups), 
                         name=f"{prefix}_shuffle")(x)
    
    # Depthwise conv 3x3
    x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same',
                                use_bias=False, name=f"{prefix}_dwconv")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    
    # Group conv 1x1 (no activation before add)
    final_channels = out_channels if strides == 1 else out_channels - in_channels
    x = layers.Conv2D(final_channels, (1, 1), padding='same',
                      use_bias=False, name=f"{prefix}_gconv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn3")(x)
    
    # Residual connection
    if strides == 1:
        x = layers.Add(name=f"{prefix}_add")([x, inputs])
    else:
        avg = layers.AveragePooling2D(pool_size=3, strides=2, padding='same',
                                       name=f"{prefix}_avgpool")(inputs)
        x = layers.Concatenate(name=f"{prefix}_concat")([x, avg])
    
    x = layers.Activation('relu', name=f"{prefix}_relu_out")(x)
    return x


def ShuffleNet(input_shape=(32, 32, 3), classes=10, scale_factor=1.0,
               groups=1, num_shuffle_units=[3, 7, 3], 
               bottleneck_ratio=0.25) -> Model:
    """
    ShuffleNet model adapted for MNIST
    
    Args:
        input_shape: Input image shape
        classes: Number of output classes
        scale_factor: Width multiplier
        groups: Number of groups for group convolutions
        num_shuffle_units: Number of units in each stage
        bottleneck_ratio: Bottleneck ratio
    
    Returns:
        Keras Model
    """
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    
    # Calculate output channels for each stage
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp * out_dim_stage_two[groups] * scale_factor
    out_channels_in_stage[0] = 24 * scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(out_channels_in_stage[0], (3, 3), strides=(2, 2),
                      padding='same', use_bias=False, activation='relu',
                      name='conv1')(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                            name='maxpool1')(x)
    
    # Shuffle stages
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        stage_num = stage + 2
        
        for block in range(repeat + 1):
            if block == 0:
                strides = 2
                in_ch = out_channels_in_stage[stage]
            else:
                strides = 1
                in_ch = out_channels_in_stage[stage + 1]
            
            x = shuffle_unit(x, in_ch, out_channels_in_stage[stage + 1],
                            groups, bottleneck_ratio, strides, stage_num, block + 1)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    outputs = layers.Dense(classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs, name=f"ShuffleNet_{scale_factor}X")
    return model


def ShuffleNetMini(input_shape=(32, 32, 3), classes=10) -> Model:
    """
    Minimal ShuffleNet for very constrained MCUs like F446RE
    """
    return ShuffleNet(
        input_shape=input_shape,
        classes=classes,
        scale_factor=0.5,
        groups=1,
        num_shuffle_units=[2, 4, 2],
        bottleneck_ratio=0.25
    )

