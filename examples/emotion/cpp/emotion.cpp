/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file emotion.cpp
 * @brief Emotion recognition example using vision_service API.
 */

#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <sstream>     // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)
#include <fstream>     // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)

#include <opencv2/opencv.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>     // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

namespace fs = std::filesystem;

static std::string resolve_path(const fs::path& root, const std::string& path) {
    if (path.empty()) return path;
    if (path[0] == '/' || (path.size() >= 2 && path[1] == ':')) return path;
    return (root / path).lexically_normal().string();
}

std::vector<std::string> load_labels(const std::string& label_file_path) {
    std::vector<std::string> labels;
    std::ifstream file(label_file_path);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_yaml> [options]\n"
                << "  Emotion 情绪识别示例（vision_service 接口）\n"
                << "Options (any order after config_yaml):\n"
                << "  --model-path <path>   Override model_path in yaml\n"
                << "  --image <path>   Input image path\n"
                << "  --output <path>  Output image path (default: result_emotion.jpg)\n"
                << "  --help           Show this help\n"
                << "\nExample:\n"
                << "  " << program_name << " examples/emotion/config/emotion.yaml\n"
                << "  " << program_name
                << " examples/emotion/config/emotion.yaml --image face.png --output result_emotion.jpg\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "result_emotion.jpg";
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        }
    }

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
    if (image_path.empty()) {
        std::cerr << "Error: No --image and no test_image in config" << std::endl;
        return 1;
    }

    std::vector<std::string> labels;
    if (fs::exists(config_path)) {
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            fs::path root = fs::path(config_path).parent_path().parent_path().parent_path().parent_path();
            if (config["label_file_path"]) {
                std::string lp = resolve_path(root, config["label_file_path"].as<std::string>());
                labels = load_labels(lp);
            }
        } catch (...) {}
    }
    if (labels.empty()) labels = {"neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"};

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

    if (results.empty()) {
        std::cout << "No emotion result." << std::endl;
        return 0;
    }

    int emotion_class = results[0].label;
    float emotion_score = results[0].score;
    std::string emotion_name = (emotion_class >= 0 && static_cast<size_t>(emotion_class) < labels.size())
                                ? labels[static_cast<size_t>(emotion_class)] : "unknown";

    std::cout << "Emotion: " << emotion_name << " (class " << emotion_class << ", score: "
                << std::fixed << std::setprecision(4) << emotion_score << ")" << std::endl;

    std::ostringstream oss;
    oss << emotion_name << " " << std::fixed << std::setprecision(2) << emotion_score;
    cv::putText(img, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << std::endl;

    return 0;
}
