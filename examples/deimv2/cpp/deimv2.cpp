/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " <config.yaml> [options]\n"
        << "  --model-path <path>  Override model_path\n"
        << "  --image <path>       Override test_image\n"
        << "  --output <path>      Output image (default: deimv2_result.jpg)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const std::string config_path = argv[1];
    std::string model_path;
    std::string image_path;
    std::string output_path = "deimv2_result.jpg";
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model-path" && index + 1 < argc) {
            model_path = argv[++index];
        } else if (argument == "--image" && index + 1 < argc) {
            image_path = argv[++index];
        } else if (argument == "--output" && index + 1 < argc) {
            output_path = argv[++index];
        } else if (argument == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: "
                << argument << std::endl;
            return 1;
        }
    }

    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, model_path, false);
    if (!service) {
        std::cerr << "Create failed: "
            << VisionService::LastCreateError() << std::endl;
        return 1;
    }
    if (image_path.empty()) {
        image_path = service->GetDefaultImage();
    }
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Could not read image: " << image_path << std::endl;
        return 1;
    }

    VisionServiceResponse response;
    const VisionServiceStatus status = service->Infer(image, &response);
    if (status != VISION_SERVICE_OK || !response.ok) {
        std::cerr << "Infer failed: " << service->LastError() << std::endl;
        return 1;
    }

    const std::vector<std::string> class_names =
        service->GetClassNames();
    std::cout << "Detections: " << response.results.size() << std::endl;
    for (const vision::Result& result : response.results) {
        const int label = vision::get_label(result);
        const vision::BoundingBox box = vision::get_bbox(result);
        const std::string name =
            label >= 0 && static_cast<size_t>(label) < class_names.size()
            ? class_names[static_cast<size_t>(label)]
            : "class " + std::to_string(label);
        std::cout << "  " << name << " (" << label << "), score: "
            << std::fixed << std::setprecision(4)
            << vision::get_score(result) << ", box: ["
            << box.x1 << ", " << box.y1 << ", "
            << box.x2 << ", " << box.y2 << "]\n";
    }
    if (!service->SupportsDraw()) {
        std::cerr << "DEIMv2 must support Draw" << std::endl;
        return 1;
    }
    cv::Mat output;
    if (service->Draw(image, response, &output) != VISION_SERVICE_OK ||
        output.empty()) {
        std::cerr << "Draw failed: " << service->LastError() << std::endl;
        return 1;
    }
    if (!cv::imwrite(output_path, output)) {
        std::cerr << "Could not write output: " << output_path << std::endl;
        return 1;
    }
    std::cout << "Saved: " << output_path << std::endl;
    return 0;
}
