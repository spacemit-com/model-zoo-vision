/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example_emotion.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
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
    if (face_rel.empty()) {
        std::cerr << "Error: face_model required in " << app_config_path << std::endl;
        return -1;
    }
    const std::string face_cfg_abs = ResolveConfigPath(config_dir, project_root, face_rel);

    std::unique_ptr<VisionService> face_service =
        VisionService::Create(face_cfg_abs, "", false);
    if (!face_service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    // image mode: static ResNet50 classifier. Loaded only when NOT using camera.
    std::unique_ptr<VisionService> emotion_service;
    if (!use_camera) {
        const std::string emotion_rel = YamlString(app_cfg, "emotion_model");
        if (emotion_rel.empty()) {
            std::cerr << "Error: emotion_model required for image mode in "
                << app_config_path << std::endl;
            return -1;
        }
        emotion_cfg_abs = ResolveConfigPath(config_dir, project_root, emotion_rel);
        try {
            emotion_cfg = YAML::LoadFile(emotion_cfg_abs);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
        emotion_service = VisionService::Create(emotion_cfg_abs, "", false);
        if (!emotion_service) {
            std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
            return -1;
        }
    }

    // camera mode: dynamic emotion (ResNet50 features backbone + LSTM).
    // Loaded only when --use-camera is set.
    std::unique_ptr<VisionService> feature_service;
    std::unique_ptr<VisionService> lstm_service;
    if (use_camera) {
        const std::string feature_rel = YamlString(app_cfg, "feature_model");
        const std::string lstm_rel = YamlString(app_cfg, "lstm_model");
        if (feature_rel.empty() || lstm_rel.empty()) {
            std::cerr << "Error: camera mode needs feature_model and lstm_model in "
                << app_config_path << std::endl;
            return -1;
        }
        const std::string feature_cfg_abs = ResolveConfigPath(config_dir, project_root, feature_rel);
        const std::string lstm_cfg_abs = ResolveConfigPath(config_dir, project_root, lstm_rel);
        feature_service = VisionService::Create(feature_cfg_abs, "", false);
        if (!feature_service) {
            std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
            return -1;
        }
        lstm_service = VisionService::Create(lstm_cfg_abs, "", false);
        if (!lstm_service) {
            std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
            return -1;
        }
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
        emotion_labels = {
            "neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"};
    }

    // Faces smaller than this (min of bbox width/height, px) skip emotion recognition.
    const int min_face_size = app_cfg["min_face_size"] ? app_cfg["min_face_size"].as<int>() : 0;

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

            // Skip too-small faces: emotion is unreliable. Draw a thin gray box.
            if (std::min(x2 - x1, y2 - y1) < min_face_size) {
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(128, 128, 128), 1);
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

        // Dynamic emotion: per-frame ResNet50 feature (512-d) accumulated into a
        // 10-frame sliding window, then classified by the LSTM via InferSequence.
        constexpr int kSeqLen = 10;
        constexpr int kFeatureDim = 512;
        std::deque<std::vector<float>> feat_buffer;

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) {
                continue;
            }
            ++frame_count;
            cv::Mat vis = frame.clone();

            // Per-frame work in a lambda so any early-out still falls through to the
            // unified imshow / waitKey / save handling at the loop tail.
            [&]() {
                std::vector<VisionServiceResult> face_results;
                const VisionServiceStatus ret = face_service->InferImage(frame, &face_results);
                if (ret != VISION_SERVICE_OK || face_results.empty()) {
                    feat_buffer.clear();  // drop window on lost track
                    if (frame_count <= 5 || frame_count % 30 == 0) {
                        std::cout << "Frame " << frame_count << ": no face detected" << std::endl;
                    }
                    return;
                }

                // Filter by size first, then pick the highest-confidence eligible face.
                // This avoids a distant high-score small face out-scoring a near large
                // one and causing the whole frame to be skipped.
                const VisionServiceResult* best = nullptr;
                bool saw_small = false;
                for (const auto& r : face_results) {
                    const int rx1 = static_cast<int>(std::max(0.f, r.x1));
                    const int ry1 = static_cast<int>(std::max(0.f, r.y1));
                    const int rx2 = static_cast<int>(std::min(static_cast<float>(frame.cols), r.x2));
                    const int ry2 = static_cast<int>(std::min(static_cast<float>(frame.rows), r.y2));
                    if (rx2 <= rx1 || ry2 <= ry1) {
                        continue;
                    }
                    if (std::min(rx2 - rx1, ry2 - ry1) < min_face_size) {
                        saw_small = true;
                        cv::rectangle(vis, cv::Point(rx1, ry1), cv::Point(rx2, ry2),
                            cv::Scalar(128, 128, 128), 1);
                        continue;
                    }
                    if (best == nullptr || r.score > best->score) {
                        best = &r;
                    }
                }

                if (best == nullptr) {
                    // No eligible face (none, or all too small): drop the window.
                    feat_buffer.clear();
                    if (saw_small) {
                        cv::putText(vis, "face too small", cv::Point(10, 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128, 128, 128), 1);
                    }
                    return;
                }

                const int x1 = static_cast<int>(std::max(0.f, best->x1));
                const int y1 = static_cast<int>(std::max(0.f, best->y1));
                const int x2 = static_cast<int>(std::min(static_cast<float>(frame.cols), best->x2));
                const int y2 = static_cast<int>(std::min(static_cast<float>(frame.rows), best->y2));

                const cv::Mat face_roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                std::vector<float> feat;
                if (feature_service->InferEmbedding(face_roi, &feat) != VISION_SERVICE_OK
                    || feat.size() != static_cast<size_t>(kFeatureDim)) {
                    cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                    return;
                }
                feat_buffer.push_back(std::move(feat));
                while (feat_buffer.size() > static_cast<size_t>(kSeqLen)) {
                    feat_buffer.pop_front();
                }

                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                std::ostringstream oss;
                if (feat_buffer.size() < static_cast<size_t>(kSeqLen)) {
                    oss << "buffering " << feat_buffer.size() << "/" << kSeqLen;
                } else {
                    std::vector<float> flat;
                    flat.reserve(static_cast<size_t>(kSeqLen) * kFeatureDim);
                    for (const auto& f : feat_buffer) {
                        flat.insert(flat.end(), f.begin(), f.end());
                    }
                    std::vector<float> probs;
                    if (lstm_service->InferSequence(flat.data(), 0, 0, &probs) == VISION_SERVICE_OK
                        && !probs.empty()) {
                        const int cls = static_cast<int>(
                            std::max_element(probs.begin(), probs.end()) - probs.begin());
                        const float score = probs[static_cast<size_t>(cls)];
                        const std::string name =
                            (cls >= 0 && cls < static_cast<int>(emotion_labels.size()))
                                ? emotion_labels[static_cast<size_t>(cls)] : "unknown";
                        oss << name << ": " << std::fixed << std::setprecision(2) << score;
                    } else {
                        oss << "lstm error";
                    }
                }
                cv::putText(vis, oss.str(), cv::Point(x1, std::max(y1 - 10, 20)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            }();

            // Unified display / key handling for every branch above.
            cv::imshow("Emotion", vis);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                break;
            }
            if (key == 's') {
                const std::string out = "emotion_camera_" + std::to_string(frame_count) + ".jpg";
                if (cv::imwrite(out, vis)) {
                    std::cout << "Saved: " << out << std::endl;
                }
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
