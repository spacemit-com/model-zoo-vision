# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Non-Maximum Suppression (NMS) Utilities
"""

import numpy as np
from typing import Tuple


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """
    Non-Maximum Suppression for object detection.

    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4)
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def multiclass_nms(boxes: np.ndarray,
                   classes: np.ndarray,
                   scores: np.ndarray,
                   iou_threshold: float = 0.45) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-class Non-Maximum Suppression.

    Performs NMS for each class separately.

    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4)
        classes: Class indices, shape (N,)
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Tuple of (filtered_boxes, filtered_classes, filtered_scores)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    unique_classes = np.unique(classes)

    keep_boxes = []
    keep_classes = []
    keep_scores = []

    for c in unique_classes:
        # Get boxes for this class
        class_mask = classes == c
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        # Apply NMS
        keep_indices = nms(class_boxes, class_scores, iou_threshold)

        if len(keep_indices) > 0:
            keep_boxes.append(class_boxes[keep_indices])
            keep_classes.append(np.full(len(keep_indices), c))
            keep_scores.append(class_scores[keep_indices])

    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Concatenate results
    final_boxes = np.concatenate(keep_boxes, axis=0)
    final_classes = np.concatenate(keep_classes, axis=0)
    final_scores = np.concatenate(keep_scores, axis=0)

    return final_boxes, final_classes, final_scores


def soft_nms(boxes: np.ndarray,
            scores: np.ndarray,
            iou_threshold: float = 0.3,
            sigma: float = 0.5,
            score_threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft Non-Maximum Suppression.

    Instead of completely removing overlapping boxes, reduces their scores.

    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4)
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold
        sigma: Gaussian parameter for score decay
        score_threshold: Minimum score threshold

    Returns:
        Tuple of (kept_boxes, kept_scores)
    """
    N = boxes.shape[0]
    if N == 0:
        return np.array([]), np.array([])

    # Make a copy to avoid modifying original
    boxes = boxes.copy()
    scores = scores.copy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    indices = np.arange(N)

    while len(indices) > 0:
        # Get box with highest score
        idx = np.argmax(scores[indices])
        i = indices[idx]
        keep.append(i)

        if len(indices) == 1:
            break

        # Remove selected box from indices
        indices = np.delete(indices, idx)

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[indices])
        yy1 = np.maximum(y1[i], y1[indices])
        xx2 = np.minimum(x2[i], x2[indices])
        yy2 = np.minimum(y2[i], y2[indices])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[indices] - inter)

        # Decay scores based on IoU
        weight = np.exp(-(ovr ** 2) / sigma)
        scores[indices] *= weight

        # Keep boxes above threshold
        valid_mask = scores[indices] >= score_threshold
        indices = indices[valid_mask]

    keep = np.array(keep, dtype=np.int32)
    return boxes[keep], scores[keep]

