/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <string>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "vision_service.h"  // NOLINT(build/include_subdir)

int main(int argc, char** argv) {
    const bool help = argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
    if (help || (argc != 2 && argc != 3 && argc != 4 && argc != 8)) {
        std::cout
            << "Usage: mobilesam1 config [image [output [x1 y1 x2 y2]]]\n"
            << "Defaults: image=test_image from YAML, "
            << "output=mobilesam1_result.jpg, box=190 70 460 280\n";
        return help ? 0 : 1;
    }
    try {
        auto service = VisionService::Create(argv[1], "", true);
        if (!service)
            throw std::runtime_error(VisionService::LastCreateError());
        const std::string image_path =
            argc >= 3 ? argv[2] : service->GetDefaultImage();
        const std::string output_path =
            argc >= 4 ? argv[3] : "mobilesam1_result.jpg";
        if (image_path.empty()) {
            throw std::runtime_error(
                "No input image: set test_image in YAML or pass image");
        }
        VisionServiceRequest request{};
        request.image = cv::imread(image_path);
        if (request.image.empty()) {
            throw std::runtime_error("Could not read image: " + image_path);
        }
        request.point_coords = {{190.0F, 70.0F}, {460.0F, 280.0F}};
        if (argc == 8) {
            request.point_coords = {{std::stof(argv[4]), std::stof(argv[5])},
                                    {std::stof(argv[6]), std::stof(argv[7])}};
        }
        request.point_labels = {2, 3};
        VisionServiceResponse response;
        if (service->Infer(request, &response) != VISION_SERVICE_OK) {
            throw std::runtime_error(service->LastError());
        }
        const auto& mask =
            std::get<vision::Segmentation>(response.results.at(0));
        cv::Mat overlay = request.image.clone();
        overlay.setTo(cv::Scalar(30, 144, 144), *mask.mask);
        cv::Mat rendered;
        cv::addWeighted(request.image, 0.5, overlay, 0.5, 0, rendered);
        cv::rectangle(rendered, request.point_coords[0],
            request.point_coords[1], cv::Scalar(0, 255, 0), 2);
        if (!cv::imwrite(output_path, rendered)) {
            throw std::runtime_error("Could not save result");
        }
        std::cout << "Score: " << mask.score
            << ", mask pixels: " << cv::countNonZero(*mask.mask) << '\n';
        std::cout << "Result: " << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
