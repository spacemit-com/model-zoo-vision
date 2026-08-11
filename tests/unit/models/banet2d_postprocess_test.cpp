/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>

#include "banet2d.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++failures;
    }
}

}  // namespace

int main() {
    const vision_deploy::BANetLetterbox vertical_geometry =
        vision_deploy::make_banet_letterbox(4, 2, 4, 4);
    check(vertical_geometry.resized_width == 4,
        "resized width should be 4");
    check(vertical_geometry.resized_height == 2,
        "resized height should be 2");
    check(vertical_geometry.pad_left == 0,
        "horizontal padding should be zero");
    check(vertical_geometry.pad_top == 1,
        "vertical padding should be one");

    const float values[16] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 2.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    cv::Mat model_disparity(4, 4, CV_32FC1, const_cast<float*>(values));
    cv::Mat restored = vision_deploy::restore_banet_disparity(
        model_disparity, vertical_geometry, cv::Size(4, 2));

    check(restored.type() == CV_32FC1, "restored map should be float32");
    check(restored.rows == 2 && restored.cols == 4,
        "restored map should use original dimensions");
    const float expected[8] = {
        0.0f, 2.0f, 2.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
    };
    for (int y = 0; y < restored.rows; ++y) {
        for (int x = 0; x < restored.cols; ++x) {
            const float actual = restored.at<float>(y, x);
            check(std::isfinite(actual), "restored values should be finite");
            check(
                std::abs(actual - expected[y * restored.cols + x]) < 1e-5f,
                "restored disparity should match hand-derived values");
        }
    }

    const vision_deploy::BANetLetterbox horizontal_geometry =
        vision_deploy::make_banet_letterbox(2, 4, 4, 4);
    check(horizontal_geometry.resized_width == 2,
        "pillarboxed resized width should be 2");
    check(horizontal_geometry.resized_height == 4,
        "pillarboxed resized height should be 4");
    check(horizontal_geometry.pad_left == 1,
        "pillarbox should add one column on the left");
    check(horizontal_geometry.pad_right == 1,
        "pillarbox should add one column on the right");

    const float horizontal_values[16] = {
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
        0.0f, 4.0f, 4.0f, 0.0f,
    };
    cv::Mat horizontal_model_disparity(
        4,
        4,
        CV_32FC1,
        const_cast<float*>(horizontal_values));
    const cv::Mat horizontal_restored =
        vision_deploy::restore_banet_disparity(
            horizontal_model_disparity,
            horizontal_geometry,
            cv::Size(2, 4));
    for (int y = 0; y < horizontal_restored.rows; ++y) {
        for (int x = 0; x < horizontal_restored.cols; ++x) {
            check(
                std::abs(
                    horizontal_restored.at<float>(y, x) - 4.0f) <
                    1e-5f,
                "pillarboxed disparity must scale from resized width");
        }
    }

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: BANet2D disparity geometry" << std::endl;
    return 0;
}
