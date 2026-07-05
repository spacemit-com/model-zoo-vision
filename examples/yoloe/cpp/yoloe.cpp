/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * YOLOE open-vocabulary instance segmentation example.
 * Text prompts come from the yaml default vocabulary, or --prompts "a,b,c".
 */

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

std::vector<std::string> SplitPrompts(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos) {
            out.push_back(tok.substr(a, b - a + 1));
        }
    }
    return out;
}

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <config_yaml> [options]\n"
        << "  --model-path <path> Override model_path in yaml\n"
        << "  --image <path>      Input image (overrides config test_image)\n"
        << "  --output <path>     Output image path (default: yoloe_result.jpg)\n"
        << "  --prompts \"a,b,c\"   Override the text vocabulary (open-vocabulary)\n"
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
    std::string output_path = "yoloe_result.jpg";
    std::string model_path_override;
    std::vector<std::string> prompts;

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
        } else if (arg == "--prompts" && i + 1 < argc) {
            prompts = SplitPrompts(argv[++i]);
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

    VisionServiceRequest request;
    request.image = img;
    request.prompts = prompts;  // empty -> use yaml default vocabulary

    VisionServiceResponse response;
    VisionServiceStatus ret = service->Infer(request, &response);
    if (ret != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << std::endl;
        return 1;
    }

    const std::vector<std::string> labels = service->GetClassNames();
    if (!response.results.empty()) {
        std::cout << "Detected " << response.results.size() << " instances:" << std::endl;
        for (const auto& result : response.results) {
            const vision::BoundingBox box = vision::get_bbox(result);
            const int label = vision::get_label(result);
            const std::string name =
                (label >= 0 && label < static_cast<int>(labels.size())) ? labels[label]
                    : std::to_string(label);
            std::cout << "  " << name << ", Score: " << std::fixed << std::setprecision(4)
                << vision::get_score(result) << ", Box: [" << box.x1 << "," << box.y1
                << "," << box.x2 << "," << box.y2 << "]" << std::endl;
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
        std::cout << "No instances detected" << std::endl;
        cv::imwrite(output_path, img);
    }
    return 0;
}
