/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Fall Detection Demo (YOLOv8-Pose + STGCN)
 *
 * Logic aligned with python example_fall_detection.py:
 * - Run pose estimation; select the detection with highest score (one person)
 * - Draw that person's box and keypoints
 * - Feed 30-frame keypoint sequence to STGCN for action recognition (7 classes, Fall Down)
 * - Show action name and fall probability; smooth fall decision over recent predictions
 */

#include "example_fall_detection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vision_service.h"
#include "common.h"

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

using vision_common::KeyPoint;
using vision_common::PoseResult;
using vision_common::draw_keypoints;

namespace {

// COCO 17 keypoint indices -> TSSTG 13 (same as Python COCO17_TO_TSTSGO13)
static constexpr int COCO17_TO_TSSTG13[] = {0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4};
static constexpr int STGCN_SEQUENCE_LENGTH = 30;

struct StgcnResult {
    std::string action_name = "—";
    bool is_fall = false;
    float fall_prob = 0.0f;
};

struct FallInfo {
    float aspect_ratio = 0.0f;
    bool vertical_check = false;
    bool angle_check = false;
    float height_ratio = 0.0f;
    std::vector<std::string> reasons;
};

/** Build one frame of 13 keypoints (x,y,vis) for STGCN from C API keypoints (at least 17). */
std::vector<float> keypoints_to_stgcn_frame(const VisionServiceKeypoint* kpts, int count,
                                            int image_width, int image_height) {
    (void)image_width;
    (void)image_height;
    std::vector<float> frame(static_cast<size_t>(13 * 3), 0.f);
    if (kpts == nullptr || count < 17) return frame;
    for (int i = 0; i < 13; ++i) {
        int idx = COCO17_TO_TSSTG13[i];
        frame[static_cast<size_t>(i * 3 + 0)] = kpts[idx].x;
        frame[static_cast<size_t>(i * 3 + 1)] = kpts[idx].y;
        frame[static_cast<size_t>(i * 3 + 2)] = kpts[idx].visibility;
    }
    return frame;
}

/** Flatten 30 frames of 13*3 into pts array for StgcnActionRecognizer::predict (pixel coords). */
std::vector<float> keypoint_buffer_to_pts(const std::deque<std::vector<float>>& buffer) {
    const size_t need = static_cast<size_t>(STGCN_SEQUENCE_LENGTH * 13 * 3);
    std::vector<float> pts(need, 0.f);
    size_t off = 0;
    for (const auto& frame : buffer) {
        size_t n = std::min(frame.size(), static_cast<size_t>(13 * 3));
        for (size_t i = 0; i < n; ++i) pts[off + i] = frame[i];
        off += 13 * 3;
    }
    return pts;
}

inline bool is_nan(float v) {
    return std::isnan(v);
}

cv::Point2f nan_pt() {
    float n = std::numeric_limits<float>::quiet_NaN();
    return cv::Point2f(n, n);
}

cv::Point2f avg_if_valid(const cv::Point2f& a, const cv::Point2f& b) {
    if (!is_nan(a.x) && !is_nan(b.x)) return (a + b) * 0.5f;
    if (!is_nan(a.x)) return a;
    if (!is_nan(b.x)) return b;
    return nan_pt();
}

bool detect_fall(const std::vector<KeyPoint>& keypoints, const PoseResult& r, float kp_threshold, FallInfo& info_out) {
    if (keypoints.size() < 17) return false;

    // COCO indices
    constexpr int NOSE = 0;
    constexpr int LEFT_SHOULDER = 5;
    constexpr int RIGHT_SHOULDER = 6;
    constexpr int LEFT_HIP = 11;
    constexpr int RIGHT_HIP = 12;
    constexpr int LEFT_ANKLE = 15;
    constexpr int RIGHT_ANKLE = 16;

    std::vector<cv::Point2f> kp(17, nan_pt());
    for (int i = 0; i < 17; ++i) {
        if (keypoints[i].visibility >= kp_threshold) {
            kp[i] = cv::Point2f(keypoints[i].x, keypoints[i].y);
        }
    }

    cv::Point2f nose = kp[NOSE];
    cv::Point2f shoulder_center = avg_if_valid(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]);
    cv::Point2f hip_center = avg_if_valid(kp[LEFT_HIP], kp[RIGHT_HIP]);
    cv::Point2f ankle_center = avg_if_valid(kp[LEFT_ANKLE], kp[RIGHT_ANKLE]);

    float bbox_width = r.bbox.x2 - r.bbox.x1;
    float bbox_height = r.bbox.y2 - r.bbox.y1;
    float aspect_ratio = (bbox_height > 0.0f) ? (bbox_width / bbox_height) : 0.0f;

    bool vertical_check = false;
    if (!is_nan(nose.y) && !is_nan(shoulder_center.y)) {
        vertical_check = nose.y > shoulder_center.y;  // nose below shoulders
    }

    bool angle_check = false;
    if (!is_nan(shoulder_center.x) && !is_nan(hip_center.x)) {
        cv::Point2f vec = hip_center - shoulder_center;
        float norm = std::sqrt(vec.x * vec.x + vec.y * vec.y);
        if (norm > 1e-6f) {
            constexpr float kPi = 3.14159265358979323846f;
            float angle = std::atan2(std::abs(vec.x), std::abs(vec.y)) * 180.0f / kPi;
            angle_check = angle > 45.0f;
        }
    }

    float height_ratio = 0.0f;
    if (!is_nan(nose.y) && !is_nan(ankle_center.y) && bbox_height > 0.0f) {
        float head_to_feet = std::abs(nose.y - ankle_center.y);
        height_ratio = head_to_feet / bbox_height;
    }

    bool is_fall = false;
    std::vector<std::string> reasons;
    if (aspect_ratio > 1.2f) {
        is_fall = true;
        std::ostringstream oss;
        oss << "AspectRatio(" << std::fixed << std::setprecision(2) << aspect_ratio << ")";
        reasons.push_back(oss.str());
    }
    if (vertical_check && angle_check) {
        is_fall = true;
        reasons.push_back("HeadBelowShoulders+Tilt");
    }
    if (aspect_ratio > 1.0f && height_ratio > 0.0f && height_ratio < 0.6f) {
        is_fall = true;
        reasons.push_back("PoseAbnormal");
    }

    info_out.aspect_ratio = aspect_ratio;
    info_out.vertical_check = vertical_check;
    info_out.angle_check = angle_check;
    info_out.height_ratio = height_ratio;
    info_out.reasons = std::move(reasons);
    return is_fall;
}

void draw_one_pose_with_action(cv::Mat& image, const PoseResult& r, float kp_threshold,
                                const StgcnResult& stgcn_result) {
    bool is_fall = stgcn_result.is_fall;
    cv::Scalar box_color = is_fall ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::Scalar kp_color = is_fall ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);

    std::vector<PoseResult> one{r};
    draw_keypoints(image, one, kp_threshold, box_color, kp_color, 2, 4);

    std::string status = stgcn_result.action_name.empty() ? "—" : stgcn_result.action_name;
    int x1 = static_cast<int>(r.bbox.x1);
    int y1 = static_cast<int>(r.bbox.y1);
    int x2 = static_cast<int>(r.bbox.x2);
    int y2 = static_cast<int>(r.bbox.y2);

    int baseline = 0;
    cv::Size ts = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    int rx1 = std::max(0, x1);
    int ry1 = std::max(0, y1 - ts.height - baseline - 10);
    int rx2 = std::min(image.cols - 1, x1 + ts.width + 10);
    int ry2 = std::max(0, y1);
    cv::rectangle(image, cv::Point(rx1, ry1), cv::Point(rx2, ry2), box_color, -1);
    cv::putText(image, status, cv::Point(x1 + 5, y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    if (is_fall) {
        std::ostringstream oss;
        oss << "Fall Down (" << std::fixed << std::setprecision(2) << stgcn_result.fall_prob << ")";
        cv::putText(image, oss.str(), cv::Point(x1, std::min(image.rows - 5, y2 + 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    } else if (!stgcn_result.action_name.empty() && stgcn_result.action_name != "—") {
        std::ostringstream oss;
        oss << "P(fall)=" << std::fixed << std::setprecision(2) << stgcn_result.fall_prob;
        cv::putText(image, oss.str(), cv::Point(x1, std::min(image.rows - 5, y2 + 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
    }
}

}  // namespace

namespace {

constexpr const char* kDefaultAppConfig =
    "applications/fall_detection/config/fall_detection.yaml";

std::filesystem::path FindProjectRoot(const std::filesystem::path& exe_path) {
    namespace fs = std::filesystem;
    auto is_repo_root = [](const fs::path& dir) {
        return fs::exists(dir / "applications") && fs::exists(dir / "examples") &&
            fs::exists(dir / "src");
    };
    fs::path dir = exe_path;
    if (!dir.empty() && fs::is_regular_file(dir)) {
        dir = dir.parent_path();
    }
    for (int i = 0; i < 8; ++i) {
        if (is_repo_root(dir)) {
            return fs::absolute(dir);
        }
        if (!dir.has_parent_path()) {
            break;
        }
        dir = dir.parent_path();
    }
    fs::path cwd = fs::current_path();
    if (is_repo_root(cwd)) {
        return fs::absolute(cwd);
    }
    if (cwd.filename() == "build" && cwd.has_parent_path()) {
        fs::path parent = cwd.parent_path();
        if (is_repo_root(parent)) {
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

std::string ResolveUserPath(const std::filesystem::path& project_root, const std::string& path) {
    namespace fs = std::filesystem;
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

std::string ResolveUnderRoot(const std::filesystem::path& project_root, const std::string& path) {
    return ResolveUserPath(project_root, path);
}

std::string YamlString(const YAML::Node& node, const char* key) {
    if (!node[key]) {
        return "";
    }
    return node[key].as<std::string>();
}

bool LooksLikeYamlPath(const std::string& path) {
    if (path.size() < 5) {
        return false;
    }
    return path.compare(path.size() - 5, 5, ".yaml") == 0 ||
        path.compare(path.size() - 4, 4, ".yml") == 0;
}

std::string ResolveConfigPath(
    const std::filesystem::path& config_dir,
    const std::filesystem::path& project_root,
    const std::string& path) {
    namespace fs = std::filesystem;
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
    namespace fs = std::filesystem;

    const fs::path project_root_path = FindProjectRoot(
        (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path());

    std::string app_config_rel = kDefaultAppConfig;
    std::string video_path;
    bool use_camera = false;
    int camera_id = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            app_config_rel = argv[++i];
        } else if (a == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (a == "--use-camera") {
            use_camera = true;
        } else if (a == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0]
                << " [config.yaml] [--config <app.yaml>] [--video <path>]"
                << " [--use-camera] [--camera-id <id>]\n";
            return 0;
        } else if (!a.empty() && a[0] != '-') {
            if (LooksLikeYamlPath(a) && app_config_rel == kDefaultAppConfig) {
                app_config_rel = a;
            } else {
                std::cerr << "Unknown argument: " << a << "\n";
                return -1;
            }
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return -1;
        }
    }

    const fs::path app_config_path = fs::path(ResolveUserPath(project_root_path, app_config_rel));
    if (!fs::exists(app_config_path)) {
        std::cerr << "Error: config not found: " << app_config_path << std::endl;
        return -1;
    }
    const fs::path config_dir = app_config_path.parent_path();

    YAML::Node app_cfg;
    try {
        app_cfg = YAML::LoadFile(app_config_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    const float kp_threshold =
        app_cfg["kp_threshold"] ? app_cfg["kp_threshold"].as<float>() : 0.3f;

    if (!use_camera) {
        if (video_path.empty()) {
            const std::string test_video = YamlString(app_cfg, "test_video");
            if (test_video.empty()) {
                std::cerr << "Error: --video not provided and test_video missing in "
                    << app_config_path << std::endl;
                return -1;
            }
            video_path = ResolveUnderRoot(project_root_path, test_video);
        } else {
            video_path = ResolveUnderRoot(project_root_path, video_path);
        }
    } else if (app_cfg["camera_id"]) {
        camera_id = app_cfg["camera_id"].as<int>();
    }

    const std::string pose_rel = YamlString(app_cfg, "pose_model");
    const std::string stgcn_rel = YamlString(app_cfg, "stgcn_model");
    if (pose_rel.empty() || stgcn_rel.empty()) {
        std::cerr << "Error: pose_model and stgcn_model required in "
            << app_config_path << std::endl;
        return -1;
    }
    const std::string pose_cfg_abs = ResolveConfigPath(config_dir, project_root_path, pose_rel);
    const std::string stgcn_cfg_abs = ResolveConfigPath(config_dir, project_root_path, stgcn_rel);

    try {
        std::unique_ptr<VisionService> pose_service =
            VisionService::Create(pose_cfg_abs, "", false);
        if (!pose_service) {
            std::string msg = "Pose model create failed";
            const std::string& err = VisionService::LastCreateError();
            if (!err.empty()) {
                msg += ": " + err;
            }
            throw std::runtime_error(msg);
        }

        std::unique_ptr<VisionService> stgcn_service =
            VisionService::Create(stgcn_cfg_abs, "", false);
        if (!stgcn_service) {
            std::string msg = "STGCN model create failed";
            const std::string& err = VisionService::LastCreateError();
            if (!err.empty()) {
                msg += ": " + err;
            }
            throw std::runtime_error(msg);
        }
        const int fall_down_class_index = stgcn_service->GetFallDownClassIndex();
        if (fall_down_class_index < 0) {
            throw std::runtime_error("STGCN model does not support fall-down class index");
        }

        const int stgcn_wait_frames = app_cfg["stgcn_wait_frames"]
            ? app_cfg["stgcn_wait_frames"].as<int>() : 10;
        const int stgcn_smooth_window = app_cfg["stgcn_smooth_window"]
            ? app_cfg["stgcn_smooth_window"].as<int>() : 5;

        std::deque<std::vector<float>> keypoint_buffer;
        int stgcn_infer_step = 0;
        std::vector<int> pred_class_hist;
        StgcnResult last_stgcn_result;

        cv::VideoCapture cap;
        if (use_camera) {
            cap.open(camera_id);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open camera: " << camera_id << std::endl;
                return -1;
            }
        } else {
            if (!std::filesystem::exists(video_path)) {
                std::cerr << "Error: Video file not found: " << video_path << std::endl;
                return -1;
            }
            cap.open(video_path);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open video: " << video_path << std::endl;
                return -1;
            }
        }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay_ms = 1;
    if (fps > 1e-3) delay_ms = std::max(1, static_cast<int>(1000.0 / fps));

    std::cout << "Start processing... press 'q' to quit.\n";
    std::cout << "Action/fall by STGCN (30-frame sequence), draw highest-score person only.\n";
    int frame_idx = 0;
    int last_warn_frame = -9999;
    const int warn_interval_frames = 30;
    auto t_prev = std::chrono::steady_clock::now();

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "End of stream.\n";
            break;
        }
        frame_idx++;
        auto t_now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t_now - t_prev).count();
        double current_fps = (dt > 1e-6) ? 1.0 / dt : 0.0;
        t_prev = t_now;

        cv::Mat vis = frame.clone();
        int fall_count = 0;

        std::vector<VisionServiceResult> pose_results;
        int ret = pose_service->InferImage(frame, &pose_results);
        if (ret != VISION_SERVICE_OK) {
            std::cerr << "Error: " << pose_service->LastError() << std::endl;
            cap.release();
            cv::destroyAllWindows();
            return -1;
        }

        if (!pose_results.empty()) {
            int best_idx = 0;
            for (size_t i = 1; i < pose_results.size(); ++i) {
                if (pose_results[i].score > pose_results[best_idx].score) best_idx = i;
            }
            const VisionServiceResult& cr = pose_results[best_idx];

            const std::vector<VisionServiceKeypoint>& kpts = cr.keypoints;
            if (kpts.size() >= 17) {
                std::vector<float> frame_pts = keypoints_to_stgcn_frame(
                    kpts.data(), static_cast<int>(kpts.size()), frame.cols, frame.rows);
                keypoint_buffer.push_back(std::move(frame_pts));
                if (keypoint_buffer.size() > static_cast<size_t>(STGCN_SEQUENCE_LENGTH)) {
                    keypoint_buffer.pop_front();
                }
                if (keypoint_buffer.size() == static_cast<size_t>(STGCN_SEQUENCE_LENGTH)) {
                    stgcn_infer_step++;
                    if (stgcn_infer_step % std::max(1, stgcn_wait_frames) == 0) {
                        std::vector<float> pts = keypoint_buffer_to_pts(keypoint_buffer);
                        std::vector<float> probs;
                        if (stgcn_service->InferSequence(pts.data(), frame.cols, frame.rows, &probs)
                                == VISION_SERVICE_OK
                                && probs.size() >= static_cast<size_t>(fall_down_class_index + 1)) {
                            int pred_class = static_cast<int>(
                                std::max_element(probs.begin(), probs.end()) - probs.begin());
                            pred_class_hist.push_back(pred_class);
                            if (pred_class_hist.size() > static_cast<size_t>(std::max(stgcn_smooth_window, 1))) {
                                pred_class_hist.erase(pred_class_hist.begin());
                            }
                            float fall_prob = probs[static_cast<size_t>(fall_down_class_index)];
                            bool is_fall_smooth = false;
                            if (!pred_class_hist.empty()) {
                                int fall_count_hist = 0;
                                for (int c : pred_class_hist) {
                                    if (c == fall_down_class_index) fall_count_hist++;
                                }
                                is_fall_smooth = (fall_count_hist > static_cast<int>(pred_class_hist.size()) / 2);
                            } else {
                                is_fall_smooth = (pred_class == fall_down_class_index);
                            }
                            std::vector<std::string> class_names = stgcn_service->GetSequenceClassNames();
                            last_stgcn_result.action_name =
                                (pred_class >= 0 && pred_class < static_cast<int>(class_names.size()))
                                    ? class_names[static_cast<size_t>(pred_class)] : "—";
                            last_stgcn_result.is_fall = is_fall_smooth;
                            last_stgcn_result.fall_prob = fall_prob;
                        }
                    }
                }
            }

            PoseResult best;
            best.bbox = vision_common::BoundingBox{cr.x1, cr.y1, cr.x2, cr.y2};
            best.score = cr.score;
            best.label = cr.label;
            if (!kpts.empty()) {
                best.keypoints.resize(kpts.size());
                for (size_t i = 0; i < kpts.size(); ++i) {
                    best.keypoints[i].x = kpts[i].x;
                    best.keypoints[i].y = kpts[i].y;
                    best.keypoints[i].visibility = kpts[i].visibility;
                }
            }
            draw_one_pose_with_action(vis, best, kp_threshold, last_stgcn_result);
            if (last_stgcn_result.is_fall) fall_count = 1;
        } else {
            if (!keypoint_buffer.empty()) {
                keypoint_buffer.pop_front();
            }
            if (frame_idx <= 5 || frame_idx % 30 == 0) {
                std::cout << "Frame " << frame_idx << ": no person detected" << std::endl;
            }
        }

        std::ostringstream oss_info;
        oss_info << "Action: " << last_stgcn_result.action_name << "  P(fall): "
                << std::fixed << std::setprecision(2) << last_stgcn_result.fall_prob;
        std::string info_line = oss_info.str();
        cv::putText(vis, info_line, cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
        cv::putText(vis, info_line, cv::Point(10, 28),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1);
        if (fall_count > 0) {
            cv::putText(vis, "FALL COUNT: " + std::to_string(fall_count), cv::Point(10, 58),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            if (frame_idx - last_warn_frame >= warn_interval_frames) {
                std::cout << "Warning: fall detected at frame=" << frame_idx << std::endl;
                last_warn_frame = frame_idx;
            }
        }
        char fps_buf[32];
        std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", current_fps);
        cv::putText(vis, fps_buf, cv::Point(10, 88), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2);
        cv::putText(vis, fps_buf, cv::Point(10, 88), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 1);
        cv::imshow("Fall Detection", vis);
        int key = cv::waitKey(delay_ms) & 0xFF;
        if (key == 'q') {
            std::cout << "Quit." << std::endl;
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
