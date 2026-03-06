# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Drawing Utilities for Visualization
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


def generate_colors(num_colors: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Generate a list of distinct colors for visualization.

    Args:
        num_colors: Number of colors to generate
        seed: Random seed for reproducibility

    Returns:
        List of RGB color tuples
    """
    np.random.seed(seed)
    colors = []
    for _ in range(max(num_colors, 100)):
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        colors.append(color)
    return colors


def draw_bbox(image: np.ndarray,
              bbox: List[float],
              label: str = "",
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw a single bounding box on an image.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Label text to display
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label if provided
    if label:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return image


def draw_detections(image: np.ndarray,
                    boxes: np.ndarray,
                    classes: np.ndarray,
                    scores: np.ndarray,
                    labels: Optional[List[str]] = None,
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
    """
    Draw multiple detection results on an image.

    Args:
        image: Input image
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4)
        classes: Class indices, shape (N,)
        scores: Confidence scores, shape (N,)
        labels: Optional list of class labels
        color: Default box color
        thickness: Line thickness

    Returns:
        Image with drawn detections
    """
    result = image.copy()
    if len(boxes) == 0:
        return result
    for i in range(len(boxes)):
        box = boxes[i]
        class_id = int(classes[i])
        score = scores[i]

        # Prepare label text
        if labels and class_id < len(labels):
            label_text = f"{labels[class_id]} {score:.2f}"
        else:
            label_text = f"Class {class_id} {score:.2f}"

        result = draw_bbox(result, box, label_text, color, thickness)

    return result


def draw_tracking_results(image: np.ndarray,
                          tracks: List[Dict[str, Any]],
                          labels: Optional[List[str]] = None,
                          thickness: int = 2) -> np.ndarray:
    """
    Draw tracking results on an image.

    Args:
        image: Input image
        tracks: List of track dictionaries with keys:
                'tlbr': [x1, y1, x2, y2]
                'track_id': int
                'score': float
                'class_id': int (optional)
        labels: Optional list of class labels
        thickness: Line thickness

    Returns:
        Image with drawn tracks
    """
    result = image.copy()
    if not tracks:
        return result
    colors = generate_colors(100)

    for track in tracks:
        tlbr = track['tlbr']
        track_id = track['track_id']
        score = track['score']
        class_id = track.get('class_id', 0)

        x1, y1, x2, y2 = map(int, tlbr)

        # Get color based on track_id
        color = colors[track_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        if labels and class_id < len(labels):
            label_text = f"{labels[class_id]} ID:{track_id} {score:.2f}"
        else:
            label_text = f"ID:{track_id} {score:.2f}"

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            result,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            result,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return result


def draw_keypoints(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    kp_color: Tuple[int, int, int] = (255, 0, 0),
    line_thickness: int = 2,
    kp_radius: int = 5,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    KP_CONNECTIONS = [
        [16, 14], [14, 12], [15, 13], [13, 11], [12, 11],
        [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 6], [5, 11], [6, 12],
        [11, 13], [12, 14],
        [0, 1], [0, 2], [1, 3], [2, 4],
        [0, 5], [0, 6],
        [3, 5], [4, 6],
    ]

    for det in detections:
        # 绘制检测框
        box = det['box']
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, line_thickness)

        # 绘制关键点和连接线
        keypoints = det['keypoints']
        for i, (x, y, vis) in enumerate(keypoints):
            if vis < confidence_threshold:
                continue  # 跳过不可见关键点
            cv2.circle(image, (int(x), int(y)), kp_radius, kp_color, -1)

        # 绘制关键点连接线
        for (start_idx, end_idx) in KP_CONNECTIONS:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue  # 防止越界访问
            start_x, start_y, start_vis = keypoints[start_idx]
            end_x, end_y, end_vis = keypoints[end_idx]
            if start_vis < confidence_threshold or end_vis < confidence_threshold:
                continue
            cv2.line(
                image,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                kp_color,
                line_thickness,
            )

    return image


def draw_segmentation(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    masks: Optional[np.ndarray] = None,
    labels: List[str] = None,
) -> np.ndarray:
    """
    在图像上绘制检测框和分割掩码

    Args:
        image: 原始图像 (BGR格式)
        detections: 检测结果列表，每个元素包含 bbox, class, class_id, confidence
        masks: 分割掩码数组 [N, H, W]，可选
        labels: 类别标签列表

    Returns:
        绘制后的图像
    """
    result_image = image.copy()
    orig_h, orig_w = image.shape[:2]
    src_colors = [
        (4, 42, 255), (11, 219, 235), (243, 243, 243), (0, 223, 183),
        (17, 31, 104), (255, 111, 221), (255, 68, 79), (204, 237, 0),
        (0, 243, 68), (189, 0, 255), (0, 180, 255), (221, 0, 186),
        (0, 255, 255), (38, 192, 0), (1, 255, 179),
        (125, 36, 255), (123, 0, 104), (255, 27, 108), (252, 109, 47),
        (162, 255, 11),
    ]
    # 如果有掩码，先绘制掩码
    if masks is not None and len(masks) > 0:
        # 创建掩码叠加层
        overlay = result_image.copy()

        for i, mask in enumerate(masks):
            if i >= len(detections):
                break

            # 获取类别ID和颜色
            class_id = detections[i].get('class_id', 0)
            color_idx = class_id % len(src_colors)
            color = src_colors[color_idx]

            # 将掩码转换为3通道
            mask_3d = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_3d[:, :, 0] = mask_binary * color[0]
            mask_3d[:, :, 1] = mask_binary * color[1]
            mask_3d[:, :, 2] = mask_binary * color[2]

            # 叠加掩码
            overlay = cv2.addWeighted(overlay, 1.0, mask_3d, 0.5, 0)

        result_image = overlay

    # 绘制边界框和标签
    for i, det in enumerate(detections):
        bbox = det['bbox']  # [x1, y1, x2, y2]
        class_name = labels[det.get('class_id', 0)]
        confidence = det.get('confidence', 0.0)
        class_id = det.get('class_id', 0)

        # 确保坐标是整数且在图像范围内
        x1 = max(0, min(int(bbox[0]), orig_w - 1))
        y1 = max(0, min(int(bbox[1]), orig_h - 1))
        x2 = max(x1 + 1, min(int(bbox[2]), orig_w))
        y2 = max(y1 + 1, min(int(bbox[3]), orig_h))

        # 获取颜色
        color_idx = class_id % len(src_colors)
        color = src_colors[color_idx]

        # 绘制边界框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )

        # 绘制文本背景
        text_y = max(y1, text_height + 10)
        cv2.rectangle(
            result_image,
            (x1, text_y - text_height - baseline - 5),
            (x1 + text_width, text_y + baseline),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            result_image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255) if sum(color) < 400 else (0, 0, 0),  # 根据背景色选择文本颜色
            1,
            cv2.LINE_AA
        )

    return result_image


def put_fps_text(image: np.ndarray,
                 fps: float,
                 position: Tuple[int, int] = (10, 30),
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw FPS text on an image.

    Args:
        image: Input image
        fps: Frames per second
        position: Text position (x, y)
        color: Text color

    Returns:
        Image with FPS text
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )
    return image

