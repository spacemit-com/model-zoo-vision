/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>   // NOLINT(build/include_order)
#include <fstream>     // NOLINT(build/include_order)
#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)

#include <opencv2/opencv.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>     // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

namespace {

std::string ResolveResourcePath(const std::string& path) {
    if (path.empty() || path[0] == '/' || (path.size() >= 2 && path[1] == ':')) {
        return path;
    }
    if (std::filesystem::exists(path)) return path;
    const std::string with_parent = "../" + path;
    if (std::filesystem::exists(with_parent)) return with_parent;
    return path;
}

std::string ResolveLabelFile(const std::string& path, const std::string& config_path) {
    if (path.empty()) return path;
    namespace fs = std::filesystem;
    const auto exists = [](const fs::path& p) { return !p.empty() && fs::exists(p); };
    fs::path rel(path);
    if (rel.is_absolute() && exists(rel)) return rel.string();
    if (exists(rel)) return rel.string();
    if (exists(fs::path("..") / rel)) return (fs::path("..") / rel).string();
    const fs::path config_dir = fs::path(config_path).parent_path();
    const fs::path repo_root = config_dir.parent_path().parent_path().parent_path();
    const fs::path from_repo = repo_root / rel;
    if (exists(from_repo)) return from_repo.string();
    return ResolveResourcePath(path);
}

// Load ImageNet-format labels: one entry per line, optionally prefixed by a
// WordNet ID (e.g. "n01440764 tench, Tinca tinca" -> "tench, Tinca tinca").
std::vector<std::string> LoadImagenetLabels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;
        const size_t pos = line.find(' ');
        if (pos != std::string::npos && pos + 1 < line.size()) {
            labels.push_back(line.substr(pos + 1));
        } else {
            labels.push_back(line);
        }
    }
    return labels;
}

}  // namespace

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
                label_file_path = ResolveLabelFile(
                    config["label_file_path"].as<std::string>(), config_path);
            }
        } catch (...) {}
    }
    std::vector<std::string> labels = LoadImagenetLabels(label_file_path);

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
    if (image_path.empty()) image_path = ResolveResourcePath("test_data/images/cat.jpg");

    std::cout << "Loading image: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << image_path << std::endl;
        return 1;
    }
    VisionServiceResponse response;
    VisionServiceStatus ret = service->Infer(img, &response);
    if (ret != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << std::endl;
        return 1;
    }

    if (!response.results.empty()) {
        std::cout << "Top-5 classification results:" << std::endl;
        const size_t max_results = std::min<size_t>(response.results.size(), 5);
        for (size_t i = 0; i < max_results; i++) {
            int label_id = vision::get_label(response.results[i]);
            std::string name = (label_id >= 0 && static_cast<size_t>(label_id) < labels.size())
                                ? labels[static_cast<size_t>(label_id)] : ("Class " + std::to_string(label_id));
            std::cout << "  " << (i + 1) << ". " << name
                        << " (confidence: " << std::fixed << std::setprecision(4)
                        << vision::get_score(response.results[i]) << ")" << std::endl;
        }
    } else {
        std::cout << "No classification results" << std::endl;
    }
    return 0;
}
