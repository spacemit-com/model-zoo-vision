# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Image Processing Utilities for Computer Vision
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def letterbox(image: np.ndarray,
              new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114),
              auto: bool = False,
              scale_fill: bool = False,
              scaleup: bool = True,
              stride: int = 32) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while maintaining aspect ratio.

    Args:
        image: Input image in BGR format
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to new_shape (no padding)
        scaleup: Allow scaling up (default True)
        stride: Stride for auto padding

    Returns:
        Tuple of (resized_image, scale_ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, r, (dw, dh)


def resize_image(image: np.ndarray,
                size: Tuple[int, int],
                letterbox_mode: bool = True,
                pad_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Resize image with optional letterbox mode.

    Args:
        image: Input image (RGB or BGR)
        size: Target size (width, height)
        letterbox_mode: If True, maintain aspect ratio with padding
        pad_color: Color for padding in letterbox mode

    Returns:
        Resized image
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

    iw, ih = image.size
    w, h = size

    if letterbox_mode:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, pad_color)
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)

    return np.array(new_image)


def normalize_image(image: np.ndarray,
                   mean: Optional[Tuple[float, ...]] = None,
                   std: Optional[Tuple[float, ...]] = None,
                   scale: float = 255.0) -> np.ndarray:
    """
    Normalize image with mean and std.

    Args:
        image: Input image
        mean: Mean values for each channel (if None, use 0.5)
        std: Std values for each channel (if None, use 0.5)
        scale: Scale factor to divide pixel values

    Returns:
        Normalized image
    """
    # Convert to float
    img = image.astype(np.float32)

    # Scale
    if scale > 0:
        img /= scale

    # Normalize
    if mean is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        img -= mean

    if std is not None:
        std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        img /= std

    return img




def preprocess_classification(image: np.ndarray,
                             input_shape: Tuple[int, int],
                             mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                             std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                             resize_size: Optional[Tuple[int, int]] = None,
                             center_crop: bool = True) -> np.ndarray:
    """
    Preprocess image for classification models.

    Args:
        image: Input image in BGR format
        input_shape: Target input shape (height, width)
        mean: Mean values for normalization
        std: Std values for normalization
        resize_size: Resize to this size before cropping (if None, skip)
        center_crop: Whether to center crop

    Returns:
        Preprocessed tensor ready for inference
    """
    # Convert to float
    img = image.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize if needed
    if resize_size is not None:
        img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)

    # Center crop
    if center_crop:
        h, w = img.shape[:2]
        th, tw = input_shape
        y0 = (h - th) // 2
        x0 = (w - tw) // 2
        img = img[y0:y0+th, x0:x0+tw]
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    # Split channels and convert to CHW
    channels = cv2.split(img)
    img_chw = np.stack(channels, axis=0)

    # Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)

    return img_batch

def load_labels(label_file: str) -> list:
    """Load labels from file."""
    with open(label_file, 'r') as f:
        labels = f.readlines()
    return [label.strip() for label in labels]
