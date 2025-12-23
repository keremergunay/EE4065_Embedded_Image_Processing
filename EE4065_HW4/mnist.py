"""
Custom MNIST loader wrapper providing functions expected by the homework code.
"""
import struct
import numpy as np


def load_images(path):
    """Load MNIST images from an IDX3-UBYTE file."""
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for images file")
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return data.reshape(num_images, rows, cols)


def load_labels(path):
    """Load MNIST labels from an IDX1-UBYTE file."""
    with open(path, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for labels file")
        num_items = struct.unpack('>I', f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8).copy()
        return data


# Aliases for compatibility with 10-9.py imports
test_images = load_images
test_labels = load_labels

