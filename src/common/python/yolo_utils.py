# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLO Detection Utilities
Common functions used across YOLO model variants
"""

import numpy as np
import cv2
from typing import Tuple


def scale_boxes_letterbox(boxes: np.ndarray,
                          ratio: float,
                          pad: Tuple[float, float],
                          orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    Scale boxes from letterboxed image back to original image size.

    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4)
        ratio: Scale ratio used during letterbox preprocessing
        pad: Padding (pad_w, pad_h) applied during letterbox
        orig_shape: Original image shape (height, width)

    Returns:
        Boxes scaled to original image coordinates
    """
    boxes = boxes.copy()
    dw, dh = pad

    # Remove padding and scale back
    boxes[:, 0] = (boxes[:, 0] - dw) / ratio
    boxes[:, 1] = (boxes[:, 1] - dh) / ratio
    boxes[:, 2] = (boxes[:, 2] - dw) / ratio
    boxes[:, 3] = (boxes[:, 3] - dh) / ratio

    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])

    return boxes


def filter_boxes_by_confidence(boxes: np.ndarray,
                               box_confidences: np.ndarray,
                               box_class_probs: np.ndarray,
                               conf_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter boxes by confidence threshold.

    Args:
        boxes: Box coordinates, shape (N, 4)
        box_confidences: Box objectness scores, shape (N,) or (N, 1)
        box_class_probs: Class probabilities, shape (N, num_classes)
        conf_threshold: Confidence threshold

    Returns:
        Tuple of (filtered_boxes, classes, scores)
    """
    box_confidences = box_confidences.reshape(-1)

    # Get max class score and class index
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    # Filter by threshold (objectness * class_score)
    _class_pos = np.where(class_max_score * box_confidences >= conf_threshold)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def flatten_yolo_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Flatten YOLO output tensor from (batch, channels, h, w) to (batch*h*w, channels).

    Args:
        tensor: Input tensor of shape (B, C, H, W)

    Returns:
        Flattened tensor of shape (B*H*W, C)
    """
    return tensor.transpose(0, 2, 3, 1).reshape(-1, tensor.shape[1])


def process_box_dfl(position: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
    """
    Process DFL (Distribution Focal Loss) box predictions.
    Converts from distribution format to bounding boxes.

    Args:
        position: Position tensor from YOLO DFL head, shape (1, 4*reg_max, H, W)
        input_shape: Model input shape (height, width)

    Returns:
        Bounding boxes in xyxy format, shape (1, 4, H, W)
    """
    from .activations import dfl

    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)

    stride = np.array([
        input_shape[0] // grid_h,
        input_shape[1] // grid_w
    ]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def preprocess_yolo_letterbox(image: np.ndarray,
                              input_shape: Tuple[int, int],
                              pad_color: Tuple[int, int, int] = (114, 114, 114),
                              bgr2rgb: bool = True) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Preprocess image for YOLO models with letterbox padding.

    Args:
        image: Input image in BGR format
        input_shape: Target shape (height, width)
        pad_color: Padding color (R, G, B) or (B, G, R)
        bgr2rgb: Whether to convert BGR to RGB

    Returns:
        Tuple of (preprocessed_tensor, ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]

    # Calculate scale ratio
    r = min(input_shape[0] / shape[0], input_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = input_shape[1] - new_unpad[0], input_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Color conversion
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=pad_color)

    # Normalize and transpose
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image, r, (dw, dh)

