/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include "vision_service.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
        << " <config.yaml> [video]\n";
        return 1;
    }
    auto service = VisionService::Create(argv[1], "", true);
    if (!service) {
        std::cerr << VisionService::LastCreateError() << '\n';
        return 1;
    }
    const YAML::Node config = YAML::LoadFile(argv[1]);
    const YAML::Node box = config["initial_bbox"];
    const std::string video_path =
        argc >= 3 ? argv[2] :
        service->GetConfigPathValue("test_video");
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << '\n';
        return 1;
    }
    const double fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
    const int width =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(
        "avtrack_tracking.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to create output video\n";
        return 1;
    }

    cv::Mat frame;
    int frame_index = 0;
    while (capture.read(frame)) {
        VisionServiceRequest request;
        request.image = frame;
        if (frame_index == 0) {
            request.has_initial_bbox = true;
            const float x = box["x"].as<float>();
            const float y = box["y"].as<float>();
            request.initial_bbox = {
                x, y, x + box["w"].as<float>(),
                y + box["h"].as<float>()};
        }
        VisionServiceResponse response;
        if (service->Infer(request, &response) != VISION_SERVICE_OK) {
            std::cerr << "Frame " << frame_index << ": "
            << service->LastError() << '\n';
            return 1;
        }
        cv::Mat output;
        if (service->Draw(frame, response, &output) != VISION_SERVICE_OK) {
            std::cerr << "Draw failed\n";
            return 1;
        }
        writer.write(output);
        ++frame_index;
    }
    std::cout << "Processed " << frame_index
            << " frames, output: avtrack_tracking.mp4\n";
    return frame_index > 0 ? 0 : 1;
}
