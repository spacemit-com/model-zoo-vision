# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
CV Common Module - Shared utilities for computer vision tasks
"""
# ruff: noqa: F403, F405  -- star import and __all__ names are from .python

from .python import *

__all__ = [
    # Image processing
    'letterbox',
    'resize_image',
    'normalize_image',
    'preprocess_classification',
    'load_labels',
    # NMS
    'nms',
    'multiclass_nms',
    'soft_nms',
    # Coordinate utilities
    'scale_coords',
    'xyxy2xywh',
    'xywh2xyxy',
    'xyxy2xywh_center',
    'clip_boxes',
    'box_iou',
    # Activations
    'sigmoid',
    'softmax',
    'dfl',
    'normalize_vector',
    'cosine_similarity',
    'euclidean_distance',
    # Drawing
    'generate_colors',
    'draw_bbox',
    'draw_detections',
    'draw_tracking_results',
    'draw_keypoints',
    'draw_segmentation',
    'put_fps_text',
    # YOLO utilities
    'scale_boxes_letterbox',
    'filter_boxes_by_confidence',
    'flatten_yolo_tensor',
    'process_box_dfl',
    'preprocess_yolo_letterbox',
    # Config loader
    'load_model_config',
    'get_default_params',
    'merge_params',
]
