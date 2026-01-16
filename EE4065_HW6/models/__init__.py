# CNN Models for Handwritten Digit Recognition
# EE 4065 - Embedded Digital Image Processing - Homework 6

from .squeezenet import SqueezeNet
from .efficientnet import EfficientNet
from .mobilenet import MobileNetV2
from .resnet import ResNet
from .shufflenet import ShuffleNet

__all__ = ['SqueezeNet', 'EfficientNet', 'MobileNetV2', 'ResNet', 'ShuffleNet']

