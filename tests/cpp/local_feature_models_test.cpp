/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "drawing.h"
#include "lightglue_matcher.h"
#include "superpoint_extractor.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        ++failures;
    }
}

bool close(float lhs, float rhs) {
    return std::abs(lhs - rhs) < 1e-4f;
}

}  // namespace

int main() {
    std::vector<float> scores(8 * 8, 0.0f);
    scores[0] = 1.0f;             // removed by border filtering
    scores[3 * 8 + 3] = 0.9f;     // kept
    scores[3 * 8 + 4] = 0.8f;     // suppressed by the 0.9 neighbor
    scores[6 * 8 + 6] = 0.7f;     // kept

    std::vector<float> descriptor_map(2 * 2 * 2);
    for (int i = 0; i < 4; ++i) {
        descriptor_map[i] = 3.0f;
        descriptor_map[4 + i] = 4.0f;
    }

    const vision::LocalFeatures features =
        vision_deploy::build_superpoint_features(
            scores.data(),
            descriptor_map.data(),
            8,
            8,
            2,
            2,
            2,
            2,
            1,
            1,
            16,
            4,
            "superpoint");

    check(features.keypoints.size() == 2, "expected exactly two keypoints");
    check(features.descriptor_dim == 2, "descriptor dimension should be two");
    check(features.descriptors.size() == 4, "descriptor buffer should be 2x2");
    check(close(features.keypoints[0].x, 6.0f), "first x should map with x scale");
    check(close(features.keypoints[0].y, 1.5f), "first y should map with y scale");
    check(close(features.keypoints[0].visibility, 0.9f), "first score should be 0.9");
    check(close(features.keypoints[1].x, 12.0f), "second x should map to 12");
    check(close(features.keypoints[1].y, 3.0f), "second y should map to 3");
    for (size_t i = 0; i < features.keypoints.size(); ++i) {
        check(close(features.descriptors[i * 2], 0.6f),
            "descriptor first component should be normalized");
        check(close(features.descriptors[i * 2 + 1], 0.8f),
            "descriptor second component should be normalized");
    }

    check(
        vision_deploy::validate_lightglue_features(
            features, "superpoint", 2, 2).empty(),
        "valid SuperPoint features should pass validation");

    vision::LocalFeatures invalid = features;
    invalid.feature_type = "disk";
    check(
        !vision_deploy::validate_lightglue_features(
            invalid, "superpoint", 2, 2).empty(),
        "wrong frontend type should fail validation");
    invalid = features;
    invalid.keypoints.pop_back();
    check(
        !vision_deploy::validate_lightglue_features(
            invalid, "superpoint", 2, 2).empty(),
        "wrong keypoint count should fail validation");
    invalid = features;
    invalid.descriptor_dim = 3;
    check(
        !vision_deploy::validate_lightglue_features(
            invalid, "superpoint", 2, 3).empty(),
        "descriptor buffer mismatch should fail validation");
    invalid = features;
    invalid.descriptors[0] = std::numeric_limits<float>::quiet_NaN();
    check(
        !vision_deploy::validate_lightglue_features(
            invalid, "superpoint", 2, 2).empty(),
        "NaN descriptor should fail validation");

    const float log_scores[4] = {
        std::log(0.9f), std::log(0.1f),
        std::log(0.2f), std::log(0.8f),
    };
    const std::vector<vision::FeatureMatch> matches =
        vision_deploy::filter_lightglue_matches(
            log_scores, 2, features, features, 0.5f);
    check(matches.size() == 2, "two mutual matches should survive");
    check(
        matches[0].query_index == 0 && matches[0].train_index == 0,
        "first match should be 0->0");
    check(close(matches[0].score, 0.9f), "first match score should be 0.9");
    check(
        matches[1].query_index == 1 && matches[1].train_index == 1,
        "second match should be 1->1");
    check(close(matches[1].score, 0.8f), "second match score should be 0.8");

    vision::LocalFeatures drawable_features;
    drawable_features.keypoints = {
        {0.0f, 0.0f, 0.0f},
        {8.0f, 8.0f, 0.9f},
    };
    drawable_features.descriptor_dim = 2;
    drawable_features.descriptors.assign(4, 0.0f);
    drawable_features.image_width = 16;
    drawable_features.image_height = 16;
    drawable_features.feature_type = "superpoint";
    cv::Mat canvas(16, 16, CV_8UC3, cv::Scalar(0, 0, 0));
    vision_common::draw_results(
        canvas,
        {vision::Result{drawable_features}});
    check(
        canvas.at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 0),
        "zero-visibility padded keypoint must not be drawn");
    check(
        canvas.at<cv::Vec3b>(8, 8) == cv::Vec3b(0, 255, 0),
        "visible local feature keypoint should be drawn");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: local feature model helpers" << std::endl;
    return 0;
}
