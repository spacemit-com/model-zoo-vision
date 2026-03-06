/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Fire Detection Example (火焰检测示例)
 *
 * Uses vision_service API: one detector from detector_config_path + detector_model_path override.
 * App config: applications/fire_detection/config/fire_detection.yaml.
 */

#include "example_fire_detection.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "vision_service.h"

namespace {
    namespace fs = std::filesystem;

    bool ends_with(const std::string& s, const std::string& suffix) {
        if (s.size() < suffix.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    fs::path find_project_root_from_exe(const fs::path& exe_path) {
        fs::path dir = exe_path;
        if (!dir.empty() && fs::is_regular_file(dir)) dir = dir.parent_path();
        for (int i = 0; i < 8; ++i) {
            if (fs::exists(dir / "applications" / "fire_detection" / "config")) {
                fs::path abs = fs::absolute(dir);
                if (abs.filename() == "build" && abs.has_parent_path()) {
                    return abs.parent_path();
                }
                return abs;
            }
            if (fs::exists(dir / "applications") && fs::exists(dir / "examples")) {
                fs::path abs = fs::absolute(dir);
                if (abs.filename() == "build" && abs.has_parent_path()) {
                    return abs.parent_path();
                }
                return abs;
            }
            if (!dir.has_parent_path()) break;
            dir = dir.parent_path();
        }
        fs::path cwd = fs::current_path();
        if (cwd.filename() == "build" && cwd.has_parent_path()) {
            return cwd.parent_path();
        }
        return cwd;
    }

    std::string resolve_under_root(const fs::path& project_root, const std::string& p) {
        if (p.empty()) return "";
        std::string path = p;
        if (path[0] == '~') {
            const char* home = std::getenv("HOME");
            if (home && home[0] != '\0')
                path = (path.size() == 1 || path[1] == '/') ? std::string(home) + path.substr(1) : path;
        }
        fs::path in(path);
        if (in.is_absolute()) return in.string();
        return (project_root / in).lexically_normal().string();
    }

    YAML::Node load_app_yaml(const fs::path& config_file) {
        if (!fs::exists(config_file)) {
            throw std::runtime_error("Config file not found: " + config_file.string());
        }
        return YAML::LoadFile(config_file.string());
    }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 1) {
        return -1;
    }

    const fs::path exe_path = (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path();
    fs::path project_root_path = find_project_root_from_exe(exe_path);
    const fs::path app_cfg_rel = fs::path("applications") / "fire_detection" / "config" / "fire_detection.yaml";
    fs::path app_cfg_file = project_root_path / app_cfg_rel;
    bool app_cfg_from_arg = false;
    if (argc >= 2 && argv[1] && ends_with(argv[1], ".yaml")) {
        app_cfg_file = fs::absolute(fs::path(argv[1]));
        app_cfg_from_arg = true;
        if (!fs::exists(app_cfg_file)) {
            std::cerr << "Error: config file not found: " << app_cfg_file << std::endl;
            return -1;
        }
        // app_cfg is at applications/fire_detection/config/fire_detection.yaml -> go up 3 to cv root
        fs::path p = app_cfg_file.parent_path();
        for (int i = 0; i < 3 && p.has_parent_path(); ++i) {
            p = p.parent_path();
        }
        project_root_path = p;
    }
    if (!app_cfg_from_arg) {
        for (int up = 0; up < 6 && !fs::exists(app_cfg_file) && project_root_path.has_parent_path(); ++up) {
            project_root_path = project_root_path.parent_path();
            app_cfg_file = project_root_path / app_cfg_rel;
        }
    }

    YAML::Node app_cfg;
    try {
        app_cfg = load_app_yaml(app_cfg_file);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::string image_path;
    std::string output_path = "result_fire_detection.jpg";
    std::string model_path_cli;
    bool use_camera = false;
    int camera_id = app_cfg["camera_id"] ? app_cfg["camera_id"].as<int>() : 0;
    int start_i = app_cfg_from_arg ? 2 : 1;
    for (int i = start_i; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--use-camera") {
            use_camera = true;
        } else if (a == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (a == "--model-path" && i + 1 < argc) {
            model_path_cli = argv[++i];
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
                << "Options: [image_path] [output_path] [--use-camera] [--camera-id <id>] [--model-path <path>]\n"
                << "  With --use-camera: read from camera. Press 'q' to quit.\n";
            return 0;
        } else if (!a.empty() && a[0] != '-') {
            if (image_path.empty()) {
                image_path = a;
            } else {
                output_path = a;
            }
        }
    }
    if (!image_path.empty()) {
        image_path = resolve_under_root(project_root_path, image_path);
    }
    if (!use_camera && image_path.empty() && app_cfg["test_image"]) {
        image_path = resolve_under_root(project_root_path, app_cfg["test_image"].as<std::string>());
    }
    if (!use_camera && image_path.empty()) {
        std::cout << "Usage: " << argv[0]
            << " [config_yaml] [image_path] [output_path] [--use-camera] [--camera-id <id>]"
            << std::endl;
        return -1;
    }

    std::string detector_config_path =
        app_cfg["detector_config_path"] ? app_cfg["detector_config_path"].as<std::string>() : "";
    std::string detector_model_path =
        app_cfg["detector_model_path"] ? app_cfg["detector_model_path"].as<std::string>() : "";
    if (detector_config_path.empty()) {
        std::cerr << "Error: detector_config_path is required in fire_detection.yaml" << std::endl;
        return -1;
    }
    fs::path detector_cfg_abs = (project_root_path / detector_config_path).lexically_normal();
    if (!fs::exists(detector_cfg_abs)) {
        std::cerr << "Error: Detector config not found: " << detector_cfg_abs << std::endl;
        return -1;
    }
    std::string model_override = !model_path_cli.empty()
        ? resolve_under_root(project_root_path, model_path_cli)
        : resolve_under_root(project_root_path, detector_model_path);

    std::unique_ptr<VisionService> service = VisionService::Create(
        detector_cfg_abs.string(),
        model_override,
        true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    auto run_detect_and_draw = [&](const cv::Mat& image, cv::Mat* out_vis) {
        std::vector<VisionServiceResult> results;
        int ret = service->InferImage(image, &results);
        if (ret != VISION_SERVICE_OK) {
            *out_vis = image.clone();
            return;
        }
        if (!results.empty()) {
            service->Draw(image, out_vis);
        } else {
            *out_vis = image.clone();
        }
    };

    if (use_camera) {
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << camera_id << std::endl;
            return -1;
        }
        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) continue;
            cv::Mat vis;
            run_detect_and_draw(frame, &vis);
            cv::imshow("Fire Detection", vis);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }
        cap.release();
        cv::destroyAllWindows();
    } else {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not read image: " << image_path << std::endl;
            return -1;
        }
        cv::Mat vis;
        run_detect_and_draw(image, &vis);
        if (cv::imwrite(output_path, vis)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save " << output_path << std::endl;
        }
    }

    return 0;
}
