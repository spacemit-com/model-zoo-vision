# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Coordinate Transformation Utilities
"""

import numpy as np
from typing import Tuple, Union


def scale_coords(img1_shape: Tuple[int, int],
                 coords: np.ndarray,
                 img0_shape: Tuple[int, int],
                 ratio_pad: Union[None, Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> np.ndarray:
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    Same as original YOLOv5 code: scale_coords method.

    Args:
        img1_shape: Source image shape (height, width)
        coords: Coordinates to scale, shape (N, 4) or (4,)
        img0_shape: Destination image shape (height, width)
        ratio_pad: Optional ((ratio, ratio), (pad_w, pad_h))

    Returns:
        Scaled coordinates (modified in-place)
    """
    # Same as original code
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # 处理一维或二维数组
    if coords.ndim == 1:
        coords[[0, 2]] -= pad[0]  # x padding
        coords[[1, 3]] -= pad[1]  # y padding
        coords[:4] /= gain
    else:  # 二维数组 (N, 4) 或 (N, 6)
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

    clip_boxes(coords, img0_shape)
    return coords


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format.

    Args:
        boxes: Boxes in [x1, y1, x2, y2] format, shape (N, 4)

    Returns:
        Boxes in [x, y, w, h] format
    """
    result = boxes.copy()
    result[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    result[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return result


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    将边界框从 (center_x, center_y, width, height) 转换为 (x1, y1, x2, y2)

    Args:
        x: 边界框数组 [..., 4]

    Returns:
        转换后的边界框数组
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xyxy2xywh_center(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h] (center format).

    Args:
        boxes: Boxes in [x1, y1, x2, y2] format, shape (N, 4)

    Returns:
        Boxes in [cx, cy, w, h] format
    """
    result = boxes.copy()
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
    result[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    result[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
    return result


def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Boxes in [x1, y1, x2, y2] format, shape (N, 4)
        image_shape: Image shape (height, width)

    Returns:
        Clipped boxes
    """
    if boxes.ndim == 1:
        boxes[0] = np.clip(boxes[0], 0, image_shape[1])  # x1
        boxes[1] = np.clip(boxes[1], 0, image_shape[0])  # y1
        boxes[2] = np.clip(boxes[2], 0, image_shape[1])  # x2
        boxes[3] = np.clip(boxes[3], 0, image_shape[0])  # y2
    else:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1])  # x coords
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0])  # y coords
    return boxes


def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of boxes.
    Same as original YOLOv5 code: box_iou method.

    Args:
        box1: First set of boxes, shape (N, 4) or (1, 4)
        box2: Second set of boxes, shape (M, 4)

    Returns:
        IoU matrix of shape (N, M)
    """
    # Handle single box case: convert to (1, 4)
    if box1.ndim == 1:
        box1 = box1[None, :]
    if box2.ndim == 1:
        box2 = box2[None, :]

    # Get intersection coordinates
    # Same as original code: box1[:, None].T gives (4, N, 1), box2.T gives (4, M)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, None].T  # Each is (N, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T  # Each is (M,)

    # Intersection coordinates (broadcasting: (N, 1) with (M,) -> (N, M))
    inter_x1 = np.maximum(b1_x1, b2_x1)  # (N, M)
    inter_y1 = np.maximum(b1_y1, b2_y1)  # (N, M)
    inter_x2 = np.minimum(b1_x2, b2_x2)  # (N, M)
    inter_y2 = np.minimum(b1_y2, b2_y2)  # (N, M)

    # Calculate intersection area (ensure non-negative)
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Calculate box1 and box2 areas
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # (N, 1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # (M,)

    # IoU = intersection area / (box1 area + box2 area - intersection area)
    union = area1 + area2 - inter_area
    iou = inter_area / union

    return iou
