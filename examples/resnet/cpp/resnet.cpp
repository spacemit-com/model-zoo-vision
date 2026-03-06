/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>   // NOLINT(build/include_order)
#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)

#include <opencv2/opencv.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>     // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)
#include "common/cpp/image_processing.h"  // NOLINT(build/include_order)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
                    << "  Options (any order): --model-path <path> --image <path>\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string model_path_override;
    std::string label_file_path;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        }
    }

    if (std::filesystem::exists(config_path)) {
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            if (config["label_file_path"]) {
                label_file_path = vision_common::resolve_path_for_resource(config["label_file_path"].as<std::string>());
            }
        } catch (...) {}
    }
    std::vector<std::string> labels = vision_common::load_labels_imagenet(label_file_path);

    std::unique_ptr<VisionService> service = VisionService::Create(
        config_path,
        model_path_override,
        true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    if (image_path.empty()) {
        const std::string default_image = service->GetDefaultImage();
        if (!default_image.empty()) {
            image_path = default_image;
        }
    }
    if (image_path.empty()) image_path = vision_common::resolve_path_for_resource("test_data/images/cat.jpg");
    if (image_path.empty()) image_path = "../test_data/images/cat.jpg";

    std::cout << "Loading image: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << image_path << std::endl;
        return 1;
    }
    std::vector<VisionServiceResult> results;
    VisionServiceStatus ret = service->InferImage(img, &results);
    if (ret != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << std::endl;
        return 1;
    }

    if (!results.empty()) {
        std::cout << "Classification results:" << std::endl;
        const size_t max_results = std::min<size_t>(results.size(), 5);
        for (size_t i = 0; i < max_results; i++) {
            int label_id = results[i].label;
            std::string name = (label_id >= 0 && static_cast<size_t>(label_id) < labels.size())
                                ? labels[static_cast<size_t>(label_id)] : ("Class " + std::to_string(label_id));
            std::cout << "  " << (i + 1) << ". " << name
                        << " (confidence: " << std::fixed << std::setprecision(4)
                        << results[i].score << ")" << std::endl;
        }
    } else {
        std::cout << "No classification results" << std::endl;
    }
    return 0;
}
