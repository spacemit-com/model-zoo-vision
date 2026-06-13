/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <utility>     // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)
#include <fstream>     // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <cstdio>      // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)

#include <opencv2/opencv.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>     // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

namespace {

// Resolve a resource path: absolute paths pass through; relative paths are tried
// as-is, then under "../" (handles running examples from build/ subdir).
std::string ResolveResourcePath(const std::string& path) {
    if (path.empty() || path[0] == '/' || (path.size() >= 2 && path[1] == ':')) {
        return path;
    }
    if (std::filesystem::exists(path)) return path;
    const std::string with_parent = "../" + path;
    if (std::filesystem::exists(with_parent)) return with_parent;
    return path;
}

// Load labels: one entry per non-empty line.
std::vector<std::string> LoadLabels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) labels.push_back(line);
    }
    return labels;
}

}  // namespace

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_yaml> [options]\n"
                << "  YOLOv5 手势检测示例（vision_service 接口）\n"
                << "Options (any order after config_yaml):\n"
                << "  --model-path <path>   Override model_path in yaml\n"
                << "  --image <path>        Input image path (overrides config test_image)\n"
                << "  --output <path>       Output image path (default: result_gesture.jpg)\n"
                << "  --use-camera          Use camera input\n"
                << "  --camera-id <i>       Camera device ID (default: 0)\n"
                << "  --help                Show this help\n"
                << "\nExample:\n"
                << "  " << program_name << " examples/yolov5_gesture/config/yolov5_gesture.yaml\n"
                << "  " << program_name
                << " examples/yolov5_gesture/config/yolov5_gesture.yaml --image test.jpg --output result_gesture.jpg\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string output_path = "result_gesture.jpg";
    std::string model_path_override;
    bool use_camera = false;
    int camera_id = 0;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
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

    std::vector<std::string> labels;
    if (std::filesystem::exists(config_path)) {
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            if (config["label_file_path"]) {
                std::string lp = ResolveResourcePath(config["label_file_path"].as<std::string>());
                labels = LoadLabels(lp);
                if (labels.empty()) {
                    std::cerr << "Warning: failed to load labels from: " << lp << std::endl;
                }
            }
        } catch (...) {}
    }

    if (use_camera) {
        std::cout << "Using camera " << camera_id << "..." << std::endl;
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << camera_id << std::endl;
            return 1;
        }
        std::cout << "Real-time YOLOv5 gesture detection. Press 'q' to quit, 's' to save." << std::endl;
        cv::Mat frame;
        int frame_count = 0;
        double t_prev = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
        double fps = 0.0;
        while (cap.read(frame)) {
            frame_count++;
            VisionServiceResponse response;
            VisionServiceStatus ret = service->Infer(frame, &response);
            if (ret != VISION_SERVICE_OK) {
                std::cerr << "Error: " << service->LastError() << std::endl;
                cap.release();
                cv::destroyAllWindows();
                return 1;
            }
            cv::Mat vis;
            if (!response.results.empty()) {
                if (frame_count <= 5 || frame_count % 30 == 0)
                    std::cout << "Frame " << frame_count << ": " << response.results.size() << " gesture(s)" << std::endl;
                auto draw_status = service->Draw(frame, response, &vis);
                if (draw_status != VISION_SERVICE_OK) {
                    std::cerr << "Draw error: " << service->LastError() << std::endl;
                    vis = frame.clone();
                }
            } else {
                vis = frame.clone();
                if (frame_count <= 5 || frame_count % 30 == 0)
                    std::cout << "Frame " << frame_count << ": no gesture detected" << std::endl;
            }
            char fps_buf[32];
            std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
            cv::putText(vis, fps_buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("YOLOv5 Gesture Detection", vis);
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            if (key == 's') {
                std::string save_path = "camera_gesture_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(save_path, vis);
                std::cout << "Saved: " << save_path << std::endl;
            }
            double t_now = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
            fps = (t_now - t_prev) > 1e-6 ? 1.0 / (t_now - t_prev) : 0.0;
            t_prev = t_now;
        }
        cap.release();
        cv::destroyAllWindows();
        std::cout << "Processed " << frame_count << " frames." << std::endl;
    } else {
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

        image_path = ResolveResourcePath(image_path);
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
            std::cout << "Detected " << response.results.size() << " gesture(s):" << std::endl;
            for (const auto& r : response.results) {
                const int label = vision::get_label(r);
                const vision::BoundingBox box = vision::get_bbox(r);
                std::string class_name = (labels.size() > static_cast<size_t>(label) && label >= 0)
                                        ? labels[static_cast<size_t>(label)] : "Class " + std::to_string(label);
                std::cout << "  " << class_name << " (class " << label << ") score="
                            << std::fixed << std::setprecision(3) << vision::get_score(r)
                            << " box=[" << box.x1 << "," << box.y1 << "," << box.x2 << "," << box.y2 << "]" << std::endl;
            }
            cv::Mat vis;
            auto draw_status = service->Draw(img, response, &vis);
            if (draw_status == VISION_SERVICE_OK) {
                cv::imwrite(output_path, vis);
            } else {
                cv::imwrite(output_path, img);
            }
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cout << "No gesture detected." << std::endl;
            cv::imwrite(output_path, img);
        }
    }

    return 0;
}
