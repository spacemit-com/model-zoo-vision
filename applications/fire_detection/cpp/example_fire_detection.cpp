/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
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

constexpr const char* kDefaultAppConfig =
    "applications/fire_detection/config/fire_detection.yaml";

bool IsRepoRoot(const fs::path& dir) {
    return fs::exists(dir / "applications") && fs::exists(dir / "examples") &&
        fs::exists(dir / "src");
}

fs::path FindProjectRoot(const fs::path& exe_path) {
    fs::path dir = exe_path;
    if (!dir.empty() && fs::is_regular_file(dir)) {
        dir = dir.parent_path();
    }
    for (int i = 0; i < 8; ++i) {
        if (IsRepoRoot(dir)) {
            return fs::absolute(dir);
        }
        if (!dir.has_parent_path()) {
            break;
        }
        dir = dir.parent_path();
    }
    fs::path cwd = fs::current_path();
    if (IsRepoRoot(cwd)) {
        return fs::absolute(cwd);
    }
    if (cwd.filename() == "build" && cwd.has_parent_path()) {
        fs::path parent = cwd.parent_path();
        if (IsRepoRoot(parent)) {
            return fs::absolute(parent);
        }
    }
    return fs::absolute(cwd);
}

std::string ExpandTilde(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return path;
    }
    if (path.size() == 1 || path[1] == '/') {
        return std::string(home) + path.substr(1);
    }
    return path;
}

std::string ResolveUserPath(const fs::path& project_root, const std::string& path) {
    if (path.empty()) {
        return "";
    }
    fs::path in(ExpandTilde(path));
    if (in.is_absolute()) {
        return in.lexically_normal().string();
    }
    const fs::path cwd = fs::current_path();
    const fs::path candidates[] = {cwd / in, project_root / in};
    for (const fs::path& candidate : candidates) {
        const fs::path abs = fs::absolute(candidate).lexically_normal();
        if (fs::exists(abs)) {
            return abs.string();
        }
    }
    return fs::absolute(project_root / in).lexically_normal().string();
}

bool LooksLikeYamlPath(const std::string& path) {
    if (path.size() < 5) {
        return false;
    }
    return path.compare(path.size() - 5, 5, ".yaml") == 0 ||
        path.compare(path.size() - 4, 4, ".yml") == 0;
}

std::string ResolveUnderRoot(const fs::path& project_root, const std::string& path) {
    return ResolveUserPath(project_root, path);
}

std::string YamlString(const YAML::Node& node, const char* key) {
    if (!node[key]) {
        return "";
    }
    return node[key].as<std::string>();
}

std::string ResolveConfigPath(
    const fs::path& config_dir,
    const fs::path& project_root,
    const std::string& path) {
    if (path.empty()) {
        return "";
    }
    fs::path local = config_dir / path;
    if (fs::exists(local)) {
        return local.lexically_normal().string();
    }
    return ResolveUnderRoot(project_root, path);
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path project_root = FindProjectRoot(
        (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path());

    std::string app_config_rel = kDefaultAppConfig;
    std::string image_path;
    std::string output_path = "result_fire_detection.jpg";
    bool use_camera = false;
    int camera_id = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            app_config_rel = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--use-camera") {
            use_camera = true;
        } else if (arg == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0]
                << " [config.yaml] [--config <app.yaml>] [--image <path>] [output_path]"
                << " [--use-camera] [--camera-id <id>]\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            if (LooksLikeYamlPath(arg) && app_config_rel == kDefaultAppConfig) {
                app_config_rel = arg;
            } else if (image_path.empty()) {
                image_path = arg;
            } else {
                output_path = arg;
            }
        }
    }

    const fs::path app_config_path = fs::path(ResolveUserPath(project_root, app_config_rel));
    if (!fs::exists(app_config_path)) {
        std::cerr << "Error: config not found: " << app_config_path << std::endl;
        return -1;
    }
    YAML::Node app_cfg;
    try {
        app_cfg = YAML::LoadFile(app_config_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    const std::string model_rel = YamlString(app_cfg, "model");
    if (model_rel.empty()) {
        std::cerr << "Error: model is required in " << app_config_path << std::endl;
        return -1;
    }
    const std::string detector_cfg_abs =
        ResolveConfigPath(app_config_path.parent_path(), project_root, model_rel);

    std::unique_ptr<VisionService> service = VisionService::Create(detector_cfg_abs, "", false);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    if (!use_camera && image_path.empty()) {
        image_path = service->GetDefaultImage();
    } else if (!image_path.empty()) {
        image_path = ResolveUnderRoot(project_root, image_path);
    }

    if (!use_camera && image_path.empty()) {
        std::cerr << "Error: provide --image or set test_image in " << detector_cfg_abs << std::endl;
        return -1;
    }

    auto run_detect_and_draw = [&](const cv::Mat& image, cv::Mat* out_vis) {
        VisionServiceResponse response;
        if (service->Infer(image, &response) != VISION_SERVICE_OK) {
            *out_vis = image.clone();
            return;
        }
        if (!response.results.empty() && service->Draw(image, response, out_vis) != VISION_SERVICE_OK) {
            *out_vis = image.clone();
        } else if (response.results.empty()) {
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
            if (frame.empty()) {
                continue;
            }
            cv::Mat vis;
            run_detect_and_draw(frame, &vis);
            cv::imshow("Fire Detection", vis);
            if ((cv::waitKey(1) & 0xFF) == 'q') {
                break;
            }
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
