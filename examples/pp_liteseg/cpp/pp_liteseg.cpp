/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_yaml> [options]\n"
                << "  --model-path <path>   Override model_path in yaml\n"
                << "  --image <path>        Input image path\n"
                << "  --output <path>       Output image path (default: pp_liteseg_result.jpg)\n"
                << "  --help                Show this help message\n"
                << "\nExample:\n"
                << "  " << program_name << " examples/pp_liteseg/config/pp_liteseg.yaml\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "pp_liteseg_result.jpg";
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

    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, model_path_override, true);
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
        std::cerr << "Error: No input image. Use --image <path> or set test_image in config." << std::endl;
        return 1;
    }

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
        std::cout << "No foreground classes (or empty segmentation); nothing to draw." << std::endl;
        return 0;
    }

    if (!service->SupportsDraw()) {
        std::cerr << "Error: Model does not support Draw()" << std::endl;
        return 1;
    }

    cv::Mat vis;
    VisionServiceStatus dr = service->Draw(img, &vis);
    if (dr != VISION_SERVICE_OK) {
        std::cerr << "Error: Draw failed: " << service->LastError() << std::endl;
        return 1;
    }

    cv::imwrite(output_path, vis);
    std::cout << "Saved: " << output_path << " (" << results.size() << " mask layer(s))" << std::endl;
    return 0;
}
