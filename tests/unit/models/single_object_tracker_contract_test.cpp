/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>

#include "av_tracker.h"
#include "mixformer_tracker.h"
#include "nanotrack_tracker.h"
#include "single_object_tracker_utils.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Callable>
void require_throws(Callable&& callable, const char* message) {
    try {
        callable();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

vision_core::InferRequest image_request(bool initialize) {
    vision_core::ImageInput input;
    input.image = cv::Mat::zeros(32, 40, CV_8UC3);
    input.has_initial_bbox = initialize;
    input.initial_bbox = {2.0f, 3.0f, 18.0f, 21.0f};
    return {input, vision_core::InferIntent::kTrack, {}};
}

}  // namespace

int main() {
    try {
        const cv::Rect2f box = vision_deploy::tracking_xyxy_to_xywh(
            {2.0f, 3.0f, 8.0f, 13.0f},
            cv::Size(20, 20));
        require(
            box.x == 2.0f && box.y == 3.0f &&
                box.width == 6.0f && box.height == 10.0f,
            "xyxy to xywh conversion is incorrect");

        require_throws(
            [] {
                (void)vision_deploy::tracking_xyxy_to_xywh(
                    {1.0f, 1.0f, 1.0f, 3.0f},
                    cv::Size(20, 20));
            },
            "zero-width initial box must fail");
        require_throws(
            [] {
                (void)vision_deploy::tracking_xyxy_to_xywh(
                    {-1.0f, 1.0f, 5.0f, 3.0f},
                    cv::Size(20, 20));
            },
            "out-of-frame initial box must fail");

        const vision::BoundingBox clipped =
            vision_deploy::tracking_xywh_to_clipped_xyxy(
                cv::Rect2f(-4.0f, -3.0f, 20.0f, 15.0f),
                cv::Size(10, 8),
                1.0f);
        require(
            clipped.x1 >= 0.0f && clipped.y1 >= 0.0f &&
                clipped.x2 <= 10.0f && clipped.y2 <= 8.0f &&
                clipped.x2 > clipped.x1 && clipped.y2 > clipped.y1,
            "tracker box clipping is invalid");

        const std::vector<cv::Point2f> points =
            vision_deploy::generate_nanotrack_points(16, 2);
        require(points.size() == 4, "NanoTrack point grid size is incorrect");
        require(
            points[0] == cv::Point2f(-16.0f, -16.0f) &&
                points[1] == cv::Point2f(0.0f, -16.0f) &&
                points[2] == cv::Point2f(-16.0f, 0.0f) &&
                points[3] == cv::Point2f(0.0f, 0.0f),
            "NanoTrack point grid coordinates are incorrect");

        const float logits[8] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, std::log(3.0f), 0.0f,
        };
        const std::vector<float> probabilities =
            vision_deploy::nanotrack_foreground_probabilities(
                logits, 2);
        require(
            probabilities.size() == 4,
            "NanoTrack probability map size is incorrect");
        require(
            std::abs(probabilities[0] - 0.5f) < 1e-6f &&
                std::abs(probabilities[2] - 0.75f) < 1e-6f,
            "NanoTrack foreground softmax is incorrect");
        require_throws(
            [] {
                (void)vision_deploy::generate_nanotrack_points(0, 2);
            },
            "NanoTrack point generation must reject invalid stride");

        vision_deploy::MixFormerTracker mixformer(
            "/does/not/exist.onnx",
            1,
            true,
            "CPUExecutionProvider");
        vision_deploy::AVTracker avtrack(
            "/does/not/exist.onnx",
            1,
            true,
            "CPUExecutionProvider");
        vision_deploy::NanoTracker nanotrack(
            "/does/not/exist-search.onnx",
            "/does/not/exist-template.onnx",
            "/does/not/exist-head.onnx",
            {},
            true,
            "CPUExecutionProvider");
        const auto request = image_request(false);
        const auto mixformer_response = mixformer.Run(request);
        const auto avtrack_response = avtrack.Run(request);
        const auto nanotrack_response = nanotrack.Run(request);
        require(
            !mixformer_response.ok &&
                mixformer_response.error_message.find("initial") !=
                    std::string::npos,
            "MixFormer must reject tracking before initialization");
        require(
            !avtrack_response.ok &&
                avtrack_response.error_message.find("initial") !=
                    std::string::npos,
            "AVTrack must reject tracking before initialization");
        require(
            !nanotrack_response.ok &&
                nanotrack_response.error_message.find("initial") !=
                    std::string::npos,
            "NanoTrack must reject tracking before initialization");

        std::cout << "PASS: single object tracker contract\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
