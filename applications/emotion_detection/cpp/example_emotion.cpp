/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example_emotion.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "common/cpp/image_processing.h"
#include "vision_service.h"

namespace {
namespace fs = std::filesystem;

constexpr const char* kDefaultAppConfig =
    "applications/emotion_detection/config/emotion_detection.yaml";

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

int main(int argc, char* argv[]) {
    const fs::path project_root = FindProjectRoot(
        (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path());

    std::string app_config_rel = kDefaultAppConfig;
    std::string image_path;
    std::string output_path = "result_emotion.jpg";
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
    const fs::path config_dir = app_config_path.parent_path();

    YAML::Node app_cfg;
    YAML::Node emotion_cfg;
    std::string emotion_cfg_abs;
    try {
        app_cfg = YAML::LoadFile(app_config_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    const std::string face_rel = YamlString(app_cfg, "face_model");
    const std::string emotion_rel = YamlString(app_cfg, "emotion_model");
    if (face_rel.empty() || emotion_rel.empty()) {
        std::cerr << "Error: face_model and emotion_model required in "
            << app_config_path << std::endl;
        return -1;
    }
    const std::string face_cfg_abs = ResolveConfigPath(config_dir, project_root, face_rel);
    emotion_cfg_abs = ResolveConfigPath(config_dir, project_root, emotion_rel);
    try {
        emotion_cfg = YAML::LoadFile(emotion_cfg_abs);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::unique_ptr<VisionService> face_service =
        VisionService::Create(face_cfg_abs, "", false);
    if (!face_service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }
    std::unique_ptr<VisionService> emotion_service =
        VisionService::Create(emotion_cfg_abs, "", false);
    if (!emotion_service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    if (!use_camera && image_path.empty()) {
        image_path = emotion_service->GetDefaultImage();
    } else if (!image_path.empty()) {
        image_path = ResolveUnderRoot(project_root, image_path);
    }

    if (!use_camera && image_path.empty()) {
        std::cerr << "Error: provide --image or set test_image in " << emotion_cfg_abs << std::endl;
        return -1;
    }

    std::vector<std::string> emotion_labels;
    const std::string label_path = YamlString(emotion_cfg, "label_file_path");
    if (!label_path.empty()) {
        try {
            emotion_labels = vision_common::load_labels(
                ResolveUnderRoot(project_root, label_path));
        } catch (...) {
            emotion_labels.clear();
        }
    }
    if (emotion_labels.empty()) {
        emotion_labels = {"neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"};
    }

    int frame_count = 0;
    auto run_face_emotion_on_image = [&](const cv::Mat& image, cv::Mat* out_image, bool log_if_empty = false) {
        std::vector<VisionServiceResult> face_results;
        const VisionServiceStatus ret = face_service->InferImage(image, &face_results);
        if (ret != VISION_SERVICE_OK || face_results.empty()) {
            if (out_image) {
                *out_image = image.clone();
            }
            if (log_if_empty && face_results.empty()
                && (frame_count <= 5 || frame_count % 30 == 0)) {
                std::cout << "Frame " << frame_count << ": no face detected" << std::endl;
            }
            return;
        }
        cv::Mat vis = image.clone();
        for (const auto& r : face_results) {
            const int x1 = static_cast<int>(std::max(0.f, r.x1));
            const int y1 = static_cast<int>(std::max(0.f, r.y1));
            const int x2 = static_cast<int>(std::min(static_cast<float>(image.cols), r.x2));
            const int y2 = static_cast<int>(std::min(static_cast<float>(image.rows), r.y2));
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            const cv::Mat face_roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            if (face_roi.empty()) {
                continue;
            }

            std::vector<VisionServiceResult> emo_results;
            if (emotion_service->InferImage(face_roi, &emo_results) != VISION_SERVICE_OK
                || emo_results.empty()) {
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                continue;
            }
            const int emotion_class = emo_results[0].label;
            const float emotion_score = emo_results[0].score;
            const std::string emotion_name =
                (emotion_class >= 0 && emotion_class < static_cast<int>(emotion_labels.size()))
                    ? emotion_labels[static_cast<size_t>(emotion_class)]
                    : "unknown";
            const cv::Scalar box_color(0, 255, 0);
            cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);
            std::ostringstream oss;
            oss << emotion_name << ": " << std::fixed << std::setprecision(2) << emotion_score;
            cv::putText(vis, oss.str(), cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2);
        }
        if (out_image) {
            *out_image = vis;
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
            ++frame_count;
            cv::Mat vis;
            run_face_emotion_on_image(frame, &vis, true);
            cv::imshow("Emotion", vis);
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
        cv::Mat result_image;
        run_face_emotion_on_image(image, &result_image);
        if (cv::imwrite(output_path, result_image)) {
            std::cout << "Result saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Error: Failed to save " << output_path << std::endl;
        }
    }

    return 0;
}
