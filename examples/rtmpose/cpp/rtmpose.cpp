/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <utility>

#include "vision_service.h"

namespace {

constexpr int kInputWidth = 192;
constexpr int kInputHeight = 256;
constexpr float kBoxPadding = 1.25F;
constexpr float kPointThreshold = 0.2F;
constexpr std::array<std::pair<int, int>, 22> kSkeleton{{
    {16, 14}, {14, 12}, {15, 13}, {13, 11}, {12, 11},
    {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 6}, {5, 11},
    {6, 12}, {11, 13}, {12, 14}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {0, 5}, {0, 6}, {3, 5}, {4, 6},
}};

struct PoseAffine {
    float center_x;
    float center_y;
    float width;
    float height;

    cv::Matx23f matrix() const {
        const float scale_x = kInputWidth / width;
        const float scale_y = kInputHeight / height;
        return {scale_x, 0.0F, kInputWidth * 0.5F - center_x * scale_x, 0.0F, scale_y,
                kInputHeight * 0.5F - center_y * scale_y};
    }

    void to_image(vision::KeyPoint* point) const {
        point->x = (point->x - kInputWidth * 0.5F) * width / kInputWidth + center_x;
        point->y = (point->y - kInputHeight * 0.5F) * height / kInputHeight + center_y;
    }
};

PoseAffine make_pose_affine(const vision::BoundingBox& box) {
    const float aspect = static_cast<float>(kInputWidth) / kInputHeight;
    float width = (box.x2 - box.x1) * kBoxPadding;
    float height = (box.y2 - box.y1) * kBoxPadding;
    if (width > height * aspect) {
        height = width / aspect;
    } else {
        width = height * aspect;
    }
    return {(box.x1 + box.x2) * 0.5F, (box.y1 + box.y2) * 0.5F, width, height};
}

void usage(const char* program) {
    std::cout << "Usage: " << program << " <config_yaml> [--image path] [--bbox x1 y1 x2 y2]"
        << " [--model-path path] [--output path]\n";
}

vision::BoundingBox parse_box(const std::array<float, 4>& values, const cv::Size& size) {
    for (float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error("bbox must be finite");
    }
    if (values[0] < 0 || values[1] < 0 || values[2] > size.width || values[3] > size.height ||
        values[2] <= values[0] || values[3] <= values[1]) {
        throw std::runtime_error("bbox must be positive-area and inside image");
    }
    return {values[0], values[1], values[2], values[3]};
}

void draw_pose(cv::Mat* image, const vision::Pose& pose) {
    const cv::Scalar color(255, 0, 0);
    for (const auto& point : pose.keypoints) {
        if (point.visibility < kPointThreshold) continue;
        cv::circle(*image, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)),
            5, color, -1);
    }
    for (const auto& [start, end] : kSkeleton) {
        if (static_cast<size_t>(start) >= pose.keypoints.size() ||
            static_cast<size_t>(end) >= pose.keypoints.size()) {
            continue;
        }
        const auto& from = pose.keypoints[start];
        const auto& to = pose.keypoints[end];
        if (from.visibility < kPointThreshold || to.visibility < kPointThreshold) continue;
        cv::line(*image, cv::Point(static_cast<int>(from.x), static_cast<int>(from.y)),
            cv::Point(static_cast<int>(to.x), static_cast<int>(to.y)), color, 2);
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    try {
        const std::string config_path = argv[1];
        std::string image_path;
        std::string output_path = "rtmpose_result.jpg";
        std::string model_path;
        std::array<float, 4> box_values{};
        bool has_box = false;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--image" && i + 1 < argc) {
                image_path = argv[++i];
            } else if (option == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (option == "--model-path" && i + 1 < argc) {
                model_path = argv[++i];
            } else if (option == "--bbox" && i + 4 < argc) {
                for (float& value : box_values) value = std::stof(argv[++i]);
                has_box = true;
            } else if (option == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + option);
            }
        }
        auto service = VisionService::Create(config_path, model_path);
        if (!service) throw std::runtime_error(VisionService::LastCreateError());
        const bool default_image = image_path.empty();
        if (default_image) image_path = service->GetDefaultImage();
        if (image_path.empty()) throw std::runtime_error("no input image");
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) throw std::runtime_error("cannot read image: " + image_path);
        if (default_image && !has_box) {
            const YAML::Node test_box = YAML::LoadFile(config_path)["test_bbox"];
            if (test_box && test_box.IsSequence() && test_box.size() == 4) {
                for (size_t i = 0; i < 4; ++i) box_values[i] = test_box[i].as<float>();
                has_box = true;
            }
        }
        const vision::BoundingBox box =
            has_box ? parse_box(box_values, image.size())
                    : vision::BoundingBox{0.0F, 0.0F, static_cast<float>(image.cols),
                        static_cast<float>(image.rows)};
        const PoseAffine affine = make_pose_affine(box);
        cv::Mat warped;
        cv::warpAffine(image, warped, affine.matrix(), cv::Size(kInputWidth, kInputHeight),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        VisionServiceResponse response;
        if (service->Infer(warped, &response) != VISION_SERVICE_OK) {
            throw std::runtime_error(service->LastError());
        }
        for (auto& result : response.results) {
            auto* pose = std::get_if<vision::Pose>(&result);
            if (pose == nullptr) continue;
            pose->bbox = box;
            for (auto& point : pose->keypoints) affine.to_image(&point);
        }
        std::cout << "Poses: " << response.results.size() << '\n';
        for (const auto& result : response.results) {
            const auto* pose = std::get_if<vision::Pose>(&result);
            if (pose == nullptr) continue;
            std::cout << "  score=" << pose->score << " keypoints=" << pose->keypoints.size()
                << " bbox=[" << pose->bbox.x1 << "," << pose->bbox.y1 << "," << pose->bbox.x2
                << "," << pose->bbox.y2 << "]\n";
            draw_pose(&image, *pose);
        }
        if (!cv::imwrite(output_path, image)) {
            throw std::runtime_error("cannot save output: " + output_path);
        }
        std::cout << "Saved: " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
