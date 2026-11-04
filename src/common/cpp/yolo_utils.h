/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLO_UTILS_H
#define YOLO_UTILS_H

#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "operators/image_preprocess/image_preprocess_geometry.h"

namespace vision_common {

// Restore box edges with the actual preprocessing transform.
void scale_coords(
    float* coords,
    const vision_operators::ImagePreprocessGeometry& geometry);


/**
 * @brief DFL (Distribution Focal Loss) decoding for YOLO boxes
 * @param boxes Box tensor data
 * @param anchor_idx Anchor index
 * @param anchors Number of anchors
 * @param grid_w Grid width
 * @param scale_w Scale factor for width
 * @param scale_h Scale factor for height
 * @return Decoded box edges (x1, y1, x2, y2) in tensor coordinates
 */
std::tuple<float, float, float, float> dfl_decode(
    const float* boxes, int anchor_idx, int anchors,
    int grid_w, float scale_w, float scale_h
);

/**
 * @brief Sigmoid activation function
 * @param x Input value
 * @return Sigmoid output: 1 / (1 + exp(-x))
 */
float sigmoid(float x);

/**
 * @brief Convert bounding box from xywh to xyxy format
 * @param xywh Input box in [x, y, w, h] format
 * @param xyxy Output box in [x1, y1, x2, y2] format
 */
void xywh2xyxy(const float* xywh, float* xyxy);

/**
 * @brief Flatten YOLO output tensor from (batch, channels, h, w) to (batch*h*w, channels)
 * @param tensor Input tensor data
 * @param batch Batch size
 * @param channels Number of channels
 * @param height Height
 * @param width Width
 * @return Flattened tensor as vector of vectors
 */
std::vector<std::vector<float>> flatten_yolo_tensor(
    const float* tensor,
    int batch, int channels, int height, int width
);

/**
 * @brief Process DFL (Distribution Focal Loss) box predictions
 * @param position Position tensor from YOLO DFL head
 * @param input_shape Model input shape (height, width)
 * @param reg_max Maximum value in distribution (default: 16)
 * @return Bounding boxes in xyxy format
 */
std::vector<std::vector<float>> process_box_dfl(
    const float* position,
    const std::pair<int, int>& input_shape,
    int reg_max = 16
);

}  // namespace vision_common

#endif  // YOLO_UTILS_H

