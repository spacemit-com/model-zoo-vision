/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo_utils.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace vision_common {

std::tuple<float, float, float, float> dfl_decode(
    const float* boxes, int anchor_idx, int anchors,
    int grid_w, float scale_w, float scale_h) {
    constexpr int kDflLen = 16;
    float xywh[4] = {0, 0, 0, 0};
    const size_t stride = static_cast<size_t>(anchors);

    for (int i = 0; i < 4; i++) {
        float exp_sum = 0.0f;
        const size_t base = static_cast<size_t>(i) * kDflLen * stride + anchor_idx;
        float exp_dfl[kDflLen];
        float max_logit = boxes[base];
        for (int dfl_idx = 1; dfl_idx < kDflLen; ++dfl_idx) {
            max_logit = std::max(max_logit, boxes[base + dfl_idx * stride]);
        }

        for (int dfl_idx = 0; dfl_idx < kDflLen; dfl_idx++) {
            exp_dfl[dfl_idx] = std::exp(boxes[base + dfl_idx * stride] - max_logit);
            exp_sum += exp_dfl[dfl_idx];
        }

        for (int dfl_idx = 0; dfl_idx < kDflLen; dfl_idx++) {
            xywh[i] += (exp_dfl[dfl_idx] / exp_sum) * dfl_idx;
        }
    }

    int h_idx = anchor_idx / grid_w;
    int w_idx = anchor_idx % grid_w;

    float x1 = (w_idx - xywh[0] + 0.5f) * scale_w;
    float y1 = (h_idx - xywh[1] + 0.5f) * scale_h;
    float x2 = (w_idx + xywh[2] + 0.5f) * scale_w;
    float y2 = (h_idx + xywh[3] + 0.5f) * scale_h;

    return std::make_tuple(x1, y1, x2, y2);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void xywh2xyxy(const float* xywh, float* xyxy) {
    float x = xywh[0];
    float y = xywh[1];
    float w = xywh[2];
    float h = xywh[3];

    xyxy[0] = x - w / 2.0f;  // x1
    xyxy[1] = y - h / 2.0f;  // y1
    xyxy[2] = x + w / 2.0f;  // x2
    xyxy[3] = y + h / 2.0f;  // y2
}

void scale_coords(
    float* coords,
    const vision_operators::ImagePreprocessGeometry& geometry) {
    geometry.validate();
    coords[0] = std::clamp(geometry.restore_x(coords[0]), 0.0f, static_cast<float>(geometry.original_width));
    coords[1] = std::clamp(geometry.restore_y(coords[1]), 0.0f, static_cast<float>(geometry.original_height));
    coords[2] = std::clamp(geometry.restore_x(coords[2]), 0.0f, static_cast<float>(geometry.original_width));
    coords[3] = std::clamp(geometry.restore_y(coords[3]), 0.0f, static_cast<float>(geometry.original_height));
}

std::vector<std::vector<float>> flatten_yolo_tensor(
    const float* tensor,
    int batch, int channels, int height, int width) {
    std::vector<std::vector<float>> flattened;
    const size_t spatial_size = static_cast<size_t>(height) * width;
    flattened.reserve(static_cast<size_t>(batch) * spatial_size);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                std::vector<float> row;
                row.reserve(channels);
                const size_t pixel = static_cast<size_t>(h) * width + w;
                for (int c = 0; c < channels; ++c) {
                    const size_t offset =
                        (static_cast<size_t>(b) * channels + c) * spatial_size + pixel;
                    row.push_back(tensor[offset]);
                }
                flattened.push_back(row);
            }
        }
    }

    return flattened;
}

std::vector<std::vector<float>> process_box_dfl(
    const float* position,
    const std::pair<int, int>& input_shape,
    int reg_max) {
    // position shape: (1, 4*reg_max, grid_h, grid_w)
    int grid_h = input_shape.first;
    int grid_w = input_shape.second;

    // Create grid
    std::vector<std::vector<float>> grid;
    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            grid.push_back({static_cast<float>(w), static_cast<float>(h)});
        }
    }

    // Process DFL for each position
    const size_t spatial_size = static_cast<size_t>(grid_h) * grid_w;
    std::vector<std::vector<float>> boxes;
    boxes.reserve(spatial_size);

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            float box_xy[4] = {0, 0, 0, 0};

            // Process each coordinate (x, y, w, h)
            for (int coord = 0; coord < 4; ++coord) {
                float exp_sum = 0.0f;
                std::vector<float> exp_vals(reg_max);

                // Compute softmax
                const size_t base_idx = static_cast<size_t>(coord) * reg_max * spatial_size +
                    static_cast<size_t>(h) * grid_w + w;

                float max_val = position[base_idx];
                for (int i = 1; i < reg_max; ++i) {
                    if (position[base_idx + i * spatial_size] > max_val) {
                        max_val = position[base_idx + i * spatial_size];
                    }
                }

                for (int i = 0; i < reg_max; ++i) {
                    exp_vals[i] = std::exp(position[base_idx + i * spatial_size] - max_val);
                    exp_sum += exp_vals[i];
                }

                // Compute expected value
                for (int i = 0; i < reg_max; ++i) {
                    box_xy[coord] += (exp_vals[i] / exp_sum) * static_cast<float>(i);
                }
            }

            // Compute stride
            float stride_h = static_cast<float>(input_shape.first) / grid_h;
            float stride_w = static_cast<float>(input_shape.second) / grid_w;

            // Convert to xyxy format
            float x = (static_cast<float>(w) + 0.5f - box_xy[0]) * stride_w;
            float y = (static_cast<float>(h) + 0.5f - box_xy[1]) * stride_h;
            float x2 = (static_cast<float>(w) + 0.5f + box_xy[2]) * stride_w;
            float y2 = (static_cast<float>(h) + 0.5f + box_xy[3]) * stride_h;

            boxes.push_back({x, y, x2, y2});
        }
    }

    return boxes;
}

}  // namespace vision_common

