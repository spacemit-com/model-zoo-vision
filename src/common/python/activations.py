# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Activation Functions and Related Utilities
"""

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Args:
        x: Input array

    Returns:
        Sigmoid output
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation function.

    Args:
        x: Input array
        axis: Axis along which to compute softmax

    Returns:
        Softmax output
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dfl(position: np.ndarray, num_points: int = 16) -> np.ndarray:
    """
    Distribution Focal Loss (DFL) for YOLO models.

    Converts predicted distributions to expected values.

    Args:
        position: Input tensor with shape (n, c, h, w)
        num_points: Number of points in the distribution (c // 4)

    Returns:
        Processed positions with shape (n, 4, h, w)
    """
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num

    # Reshape to separate the 4 coordinates
    y = position.reshape(n, p_num, mc, h, w)

    # Apply softmax to get probabilities
    y_max = np.max(y, axis=2, keepdims=True)
    y_exp = np.exp(y - y_max)
    y_sum = np.sum(y_exp, axis=2, keepdims=True)
    y = y_exp / y_sum

    # Compute expected value
    acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc_matrix).sum(axis=2)

    return y


def normalize_vector(vector: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    L2 normalize a vector or batch of vectors.

    Args:
        vector: Input vector(s)
        axis: Axis along which to normalize

    Returns:
        Normalized vector(s)
    """
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / (norm + 1e-8)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score
    """
    vec1_normalized = normalize_vector(vec1)
    vec2_normalized = normalize_vector(vec2)
    return float(np.dot(vec1_normalized.flatten(), vec2_normalized.flatten()))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))
