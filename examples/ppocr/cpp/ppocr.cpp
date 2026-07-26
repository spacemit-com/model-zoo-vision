/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PP-OCRv6 text detection + recognition example.
 */

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {
void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <config_yaml> [options]\n"
        << "  --model-path <path> Override det model_path in yaml\n"
        << "  --image <path>      Input image (overrides config test_image)\n"
        << "  --output <path>     Output image path (default: ppocr_result.jpg)\n"
        << "  --help              Show this help\n";
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "ppocr_result.jpg";
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        }
    }

    std::unique_ptr<VisionService> service = VisionService::Create(config_path, model_path_override, true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    if (image_path.empty()) {
        image_path = service->GetDefaultImage();
    }
    if (image_path.empty()) {
        std::cerr << "Error: No input image. Use --image or set test_image in config." << std::endl;
        return 1;
    }

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
        std::cout << "Recognized " << response.results.size() << " text lines:" << std::endl;
        for (const auto& result : response.results) {
            const vision::Text* t = std::get_if<vision::Text>(&result);
            if (t == nullptr) {
                continue;
            }
            std::cout << "  \"" << t->text << "\"  score=" << std::fixed << std::setprecision(3)
                << t->score << "  quad=[";
            for (size_t i = 0; i < t->polygon.size(); ++i) {
                std::cout << "(" << static_cast<int>(t->polygon[i].x) << ","
                    << static_cast<int>(t->polygon[i].y) << ")";
            }
            std::cout << "]" << std::endl;
        }
        cv::Mat vis;
        if (service->Draw(img, response, &vis) != VISION_SERVICE_OK || vis.empty()) {
            std::cerr << "Draw failed: " << service->LastError() << "; saving original image"
                << std::endl;
            vis = img;
        }
        cv::imwrite(output_path, vis);
        std::cout << "Result image saved to: " << output_path << std::endl;
    } else {
        std::cout << "No text detected" << std::endl;
        cv::imwrite(output_path, img);
    }
    return 0;
}
