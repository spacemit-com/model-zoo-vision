/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>   // NOLINT(build/include_order)
#include <cstdio>      // NOLINT(build/include_order)
#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)

#include <opencv2/opencv.hpp>  // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_yaml> [options]\n"
                << "Options (any order after config_yaml):\n"
                << "  --model-path <path>   Override model_path in yaml\n"
                << "  --video <path>        Input video path (overrides config)\n"
                << "  --use-camera          Use camera input instead of video file\n"
                << "  --camera-id <i>       Camera device ID (default: 0)\n"
                << "  --help                Show this help message\n"
                << "\nDisplay: Real-time. Press 'q' to quit.\n"
                << "Example: " << program_name
                << " examples/bytetrack/config/bytetrack.yaml [--video test.mp4]\n"
                << "         " << program_name
                << " examples/bytetrack/config/bytetrack.yaml --use-camera --camera-id 0\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string video_path;
    std::string model_path_override;
    bool use_camera = false;
    int camera_id = 0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--use-camera") {
            use_camera = true;
        } else if (arg == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
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

    if (video_path.empty() && !use_camera) {
        const std::string p = service->GetConfigPathValue("test_video");
        if (!p.empty()) video_path = p;
    }
    if (video_path.empty() && !use_camera) {
        std::cerr << "Error: No input video. Use --video <path> or --use-camera or set test_video in config."
                    << std::endl;
        return 1;
    }

    cv::VideoCapture cap;
    if (use_camera) {
        cap.open(camera_id);
        std::cout << "Using camera " << camera_id << "..." << std::endl;
    } else {
        cap.open(video_path);
        std::cout << "Opening video: " << video_path << std::endl;
    }
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open " << (use_camera ? "camera" : "video") << std::endl;
        return 1;
    }

    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int delay_ms = (fps > 0) ? std::max(1, 1000 / fps) : 33;
    std::cout << "Real-time display. Press 'q' to quit." << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    double t_prev = use_camera ? (static_cast<double>(cv::getTickCount()) / cv::getTickFrequency()) : 0.0;
    double current_fps = 0.0;
    while (cap.read(frame)) {
        frame_count++;
        std::vector<VisionServiceResult> results;
        VisionServiceStatus ret = service->InferImage(frame, &results);
        if (ret != VISION_SERVICE_OK) {
            std::cerr << "Error: " << service->LastError() << std::endl;
            cap.release();
            cv::destroyAllWindows();
            return 1;
        }
        cv::Mat vis;
        if (!results.empty()) {
            service->Draw(frame, &vis);
        } else {
            vis = frame;
        }
        if (use_camera) {
            char fps_buf[32];
            std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", current_fps);
            cv::putText(vis, fps_buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("ByteTrack", vis);
        if (cv::waitKey(delay_ms) == 'q') break;
        if (use_camera) {
            double t_now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            current_fps = (t_now - t_prev) > 1e-6 ? 1.0 / (t_now - t_prev) : 0.0;
            t_prev = t_now;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Processed " << frame_count << " frames." << std::endl;
    return 0;
}
