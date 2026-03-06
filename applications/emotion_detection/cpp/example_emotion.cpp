/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Example: emotion recognition with face detection (vision_service API)
 *
 * Two VisionService instances: yolov5-face for faces, emotion model for classification per crop.
 */

#include "example_emotion.h"

#include <algorithm>
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

#include "vision_service.h"
#include "common/cpp/image_processing.h"

namespace {
    namespace fs = std::filesystem;

    bool ends_with(const std::string& s, const std::string& suffix) {
        if (s.size() < suffix.size()) return false;
        return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    fs::path find_project_root_from_exe(const fs::path& exe_path) {
        fs::path dir = exe_path;
        if (!dir.empty() && fs::is_regular_file(dir)) dir = dir.parent_path();
        const fs::path app_cfg = fs::path("applications") / "emotion_detection" / "config" / "emotion_detection.yaml";
        for (int i = 0; i < 8; ++i) {
            if (dir.empty()) break;
            fs::path cand = fs::absolute(dir);
            if (fs::exists(cand / app_cfg)) return cand;
            if (fs::exists(cand / "applications") && fs::exists(cand / "examples")) {
                if (cand.filename() == "build" && cand.has_parent_path()) return cand.parent_path();
                return cand;
            }
            if (!dir.has_parent_path()) break;
            dir = dir.parent_path();
        }
        fs::path cwd = fs::absolute(fs::current_path());
        if (fs::exists(cwd / app_cfg)) return cwd;
        if (cwd.filename() == "build" && cwd.has_parent_path()) return cwd.parent_path();
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

    std::vector<std::string> load_emotion_labels(const fs::path& project_root, const YAML::Node& app_cfg) {
        try {
            if (!app_cfg["label_file_path"]) return {};
            std::string path = resolve_under_root(project_root, app_cfg["label_file_path"].as<std::string>());
            return vision_common::load_labels(path);
        } catch (...) {
            return {};
        }
    }
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 1) return -1;

    const fs::path exe_path = (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path();
    fs::path project_root_path = find_project_root_from_exe(exe_path);
    const fs::path app_cfg_rel = fs::path("applications") / "emotion_detection" / "config" / "emotion_detection.yaml";
    fs::path app_cfg_file = project_root_path / app_cfg_rel;
    bool app_cfg_from_arg = false;
    if (argc >= 2 && argv[1] && ends_with(argv[1], ".yaml")) {
        app_cfg_file = fs::absolute(fs::path(argv[1]));
        app_cfg_from_arg = true;
        if (!fs::exists(app_cfg_file)) {
            std::cerr << "Error: config file not found: " << app_cfg_file << std::endl;
            return -1;
        }
        // app_cfg is at applications/emotion_detection/config/emotion_detection.yaml -> go up 3 to cv root
        fs::path p = app_cfg_file.parent_path();
        for (int i = 0; i < 3 && p.has_parent_path(); ++i) p = p.parent_path();
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
    std::string output_path = "result_emotion.jpg";
    std::string face_model_path_cli;
    std::string emotion_model_path_cli;
    bool use_camera = false;
    int camera_id = app_cfg["camera_id"] ? app_cfg["camera_id"].as<int>() : 0;
    int start_i = app_cfg_from_arg ? 2 : 1;
    for (int i = start_i; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--use-camera") {
            use_camera = true;
        } else if (a == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (a == "--face-model-path" && i + 1 < argc) {
            face_model_path_cli = argv[++i];
        } else if (a == "--emotion-model-path" && i + 1 < argc) {
            emotion_model_path_cli = argv[++i];
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
                << "Options: [image_path] [output_path] [--use-camera] [--camera-id <id>]\n"
                << "         [--face-model-path <path>] [--emotion-model-path <path>]\n"
                << "  With --use-camera: read from camera. Press 'q' to quit.\n";
            return 0;
        } else if (a.empty() || a[0] == '-') {
            continue;
        } else {
            if (image_path.empty()) {
                image_path = a;
            } else {
                output_path = a;
            }
        }
    }
    if (!use_camera && image_path.empty()) {
        image_path = app_cfg["test_image"]
            ? resolve_under_root(project_root_path, app_cfg["test_image"].as<std::string>())
            : "";
    }
    if (!use_camera && image_path.empty()) {
        std::cout << "Usage: " << argv[0]
            << " [config_yaml] [image_path] [output_path] [--use-camera] [--camera-id <id>]"
            << std::endl;
        return -1;
    }

    std::vector<std::string> emotion_labels = load_emotion_labels(project_root_path, app_cfg);
    if (emotion_labels.empty()) {
        emotion_labels = {"neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"};
    }

    std::string face_config_path =
        app_cfg["face_detector_config_path"] ? app_cfg["face_detector_config_path"].as<std::string>() : "";
    std::string face_model_path =
        app_cfg["face_detector_path"] ? app_cfg["face_detector_path"].as<std::string>() : "";
    std::string emotion_config_path =
        app_cfg["emotion_config_path"] ? app_cfg["emotion_config_path"].as<std::string>() : "";
    std::string emotion_model_path =
        app_cfg["emotion_model_path"] ? app_cfg["emotion_model_path"].as<std::string>() : "";
    if (face_config_path.empty() || emotion_config_path.empty()) {
        std::cerr << "Error: face_detector_config_path and emotion_config_path required in "
            "emotion_detection.yaml" << std::endl;
        return -1;
    }
    fs::path face_cfg_abs = (project_root_path / face_config_path).lexically_normal();
    fs::path emotion_cfg_abs = (project_root_path / emotion_config_path).lexically_normal();
    if (!fs::exists(face_cfg_abs)) {
        std::cerr << "Error: face config not found: " << face_cfg_abs << std::endl;
        return -1;
    }
    if (!fs::exists(emotion_cfg_abs)) {
        std::cerr << "Error: emotion config not found: " << emotion_cfg_abs << std::endl;
        return -1;
    }
    if (!face_model_path_cli.empty()) face_model_path = resolve_under_root(project_root_path, face_model_path_cli);
    else face_model_path = resolve_under_root(project_root_path, face_model_path);
    if (!emotion_model_path_cli.empty()) {
        emotion_model_path = resolve_under_root(project_root_path, emotion_model_path_cli);
    } else {
        emotion_model_path = resolve_under_root(project_root_path, emotion_model_path);
    }

    std::unique_ptr<VisionService> face_service = VisionService::Create(
        face_cfg_abs.string(),
        face_model_path,
        true);
    if (!face_service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }
    std::unique_ptr<VisionService> emotion_service = VisionService::Create(
        emotion_cfg_abs.string(),
        emotion_model_path,
        true);
    if (!emotion_service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    auto run_face_emotion_on_image = [&](const cv::Mat& image, cv::Mat* out_image) {
        std::vector<VisionServiceResult> face_results;
        VisionServiceStatus ret = face_service->InferImage(image, &face_results);
        if (ret != VISION_SERVICE_OK || face_results.empty()) {
            if (out_image) *out_image = image.clone();
            return;
        }
        cv::Mat vis = image.clone();
        for (const auto& r : face_results) {
            int x1 = static_cast<int>(std::max(0.f, r.x1));
            int y1 = static_cast<int>(std::max(0.f, r.y1));
            int x2 = static_cast<int>(std::min(static_cast<float>(image.cols), r.x2));
            int y2 = static_cast<int>(std::min(static_cast<float>(image.rows), r.y2));
            if (x2 <= x1 || y2 <= y1) continue;

            cv::Mat face_roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            if (face_roi.empty()) continue;

            std::vector<VisionServiceResult> emo_results;
            ret = emotion_service->InferImage(face_roi, &emo_results);
            if (ret != VISION_SERVICE_OK || emo_results.empty()) {
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                continue;
            }
            int emotion_class = emo_results[0].label;
            float emotion_score = emo_results[0].score;

            std::string emotion_name = (emotion_class >= 0 && emotion_class < static_cast<int>(emotion_labels.size()))
                ? emotion_labels[static_cast<size_t>(emotion_class)] : "unknown";
            cv::Scalar box_color(0, 255, 0);
            cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 2);
            std::ostringstream oss;
            oss << emotion_name << ": " << std::fixed << std::setprecision(2) << emotion_score;
            cv::putText(vis, oss.str(), cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2);
        }
        if (out_image) *out_image = vis;
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
            run_face_emotion_on_image(frame, &vis);
            cv::imshow("Emotion", vis);
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
