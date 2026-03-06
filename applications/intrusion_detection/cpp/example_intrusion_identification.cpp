/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Intrusion Identification Demo (ByteTrack + ROI)
 *
 * Logic aligned with python example:
 * - Track persons using ByteTrack (YOLO detector + tracker)
 * - Define an ROI polygon (default center rectangle; or 3 points to infer a 4th point)
 * - If a tracked person's bottom-center "foot point" is inside ROI, mark as intrusion
 */

#include "example_intrusion_identification.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "vision_service.h"
#include "common/cpp/image_processing.h"

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::optional<cv::Point> parse_point_xy(const std::string& text) {
    auto pos = text.find(',');
    if (pos == std::string::npos) return std::nullopt;
    try {
        float fx = std::stof(text.substr(0, pos));
        float fy = std::stof(text.substr(pos + 1));
        return cv::Point(static_cast<int>(fx), static_cast<int>(fy));
    } catch (...) {
        return std::nullopt;
    }
}

cv::Point clamp_point(const cv::Point& p, int w, int h) {
    int x = std::max(0, std::min(p.x, std::max(0, w - 1)));
    int y = std::max(0, std::min(p.y, std::max(0, h - 1)));
    return cv::Point(x, y);
}

// Signed area *2 (shoelace). For image coords (y down), sign convention differs, but we only
// need a consistent "clockwise" normalization like python example.
double polygon_area2(const std::vector<cv::Point>& pts) {
    if (pts.size() < 3) return 0.0;
    double a2 = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const auto& q = pts[(i + 1) % pts.size()];
        a2 += static_cast<double>(p.x) * static_cast<double>(q.y) -
                static_cast<double>(p.y) * static_cast<double>(q.x);
    }
    return a2;
}

std::vector<cv::Point> ensure_clockwise(std::vector<cv::Point> pts) {
    // Match python: if area2 > 0, reverse
    if (polygon_area2(pts) > 0.0) {
        std::reverse(pts.begin(), pts.end());
    }
    return pts;
}

std::vector<cv::Point> default_roi_polygon(int frame_w, int frame_h) {
    if (frame_w <= 0 || frame_h <= 0) {
        return {cv::Point(0, 0), cv::Point(1, 0), cv::Point(1, 1), cv::Point(0, 1)};
    }
    int cx = frame_w / 2;
    int cy = frame_h / 2;
    int half_w = static_cast<int>(frame_w * 0.45 / 2.0);
    int half_h = static_cast<int>(frame_h * 0.35 / 2.0);

    int x1 = std::max(0, cx - half_w);
    int x2 = std::min(frame_w - 1, cx + half_w);
    int y1 = std::max(0, cy - half_h);
    int y2 = std::min(frame_h - 1, cy + half_h);
    return ensure_clockwise({cv::Point(x1, y1), cv::Point(x2, y1),
                            cv::Point(x2, y2), cv::Point(x1, y2)});
}

std::vector<cv::Point> roi_from_three_points(const cv::Point& p1, const cv::Point& p2,
                                            const cv::Point& p3, int w, int h) {
    cv::Point p4(p1.x + (p3.x - p2.x), p1.y + (p3.y - p2.y));
    std::vector<cv::Point> pts = {
        clamp_point(p1, w, h),
        clamp_point(p2, w, h),
        clamp_point(p3, w, h),
        clamp_point(p4, w, h),
    };
    return ensure_clockwise(std::move(pts));
}

cv::Mat draw_roi_overlay(const cv::Mat& image, const std::vector<cv::Point>& roi_pts, double alpha = 0.22) {
    if (roi_pts.size() < 3) return image;

    cv::Mat out = image.clone();
    cv::Mat overlay = image.clone();

    std::vector<std::vector<cv::Point>> poly{roi_pts};
    cv::fillPoly(overlay, poly, cv::Scalar(0, 0, 255));
    cv::polylines(overlay, poly, true, cv::Scalar(0, 0, 255), 2);
    cv::addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0, out);

    for (size_t i = 0; i < roi_pts.size(); ++i) {
        cv::circle(out, roi_pts[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(out, std::to_string(i + 1), roi_pts[i] + cv::Point(6, -6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    return out;
}

bool point_in_roi(const cv::Point& pt, const std::vector<cv::Point>& roi_pts) {
    if (roi_pts.size() < 3) return false;
    return cv::pointPolygonTest(roi_pts,
        cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)), false) >= 0;
}

bool is_person_label(int class_id, const std::vector<std::string>& labels) {
    if (labels.empty()) {
        return class_id == 0;  // COCO convention fallback
    }
    if (class_id < 0 || class_id >= static_cast<int>(labels.size())) return false;
    return to_lower(labels[class_id]).find("person") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    auto find_project_root = [](const fs::path& exe_path) -> fs::path {
        fs::path dir = exe_path;
        if (!dir.empty() && fs::is_regular_file(dir)) dir = dir.parent_path();
        for (int i = 0; i < 8; ++i) {
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
    };

    auto resolve_under_root = [](const fs::path& project_root, const std::string& p) -> std::string {
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
    };

    const fs::path exe_path = (argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path();
    fs::path project_root_path = find_project_root(exe_path);

    // Default application config: <cv>/applications/intrusion_detection/config/intrusion_detection.yaml
    const fs::path config_rel = fs::path("applications") / "intrusion_detection" / "config"
        / "intrusion_detection.yaml";
    fs::path config_file = project_root_path / config_rel;
    for (int up = 0; up < 6 && !fs::exists(config_file) && project_root_path.has_parent_path(); ++up) {
        project_root_path = project_root_path.parent_path();
        config_file = project_root_path / config_rel;
    }
    std::string video_path;
    bool use_camera = false;
    int camera_id = 0;
    std::string tracker_model_path_cli;

    int start_i = 1;
    if (argc >= 2 && argv[1] && ends_with(argv[1], ".yaml")) {
        config_file = argv[1];
        start_i = 2;
    }
    // Minimal CLI (yaml-first)
    for (int i = start_i; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (a == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (a == "--use-camera") {
            use_camera = true;
        } else if (a == "--camera-id" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (a == "--model-path" && i + 1 < argc) {
            tracker_model_path_cli = argv[++i];
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0]
                        << " [config_yaml] [--config <yaml>] [--video <path>] [--use-camera]"
                        << " [--camera-id <id>] [--model-path <path>]\n";
            std::cout << "  Default config: applications/intrusion_detection/config/intrusion_detection.yaml\n";
            std::cout << "  Example: " << argv[0]
                        << " applications/intrusion_detection/config/intrusion_detection.yaml\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            std::cerr << "Run with --help to see usage.\n";
            return -1;
        }
    }

    // If user passed relative config path, prefer resolving from current working directory
    // so that paths like ../applications/intrusion_detection/config/... work when run from model_zoo/cv.
    if (!config_file.is_absolute()) {
        fs::path from_cwd = fs::absolute(fs::current_path() / config_file).lexically_normal();
        if (fs::exists(from_cwd)) {
            config_file = from_cwd;
        } else {
            config_file = fs::path(resolve_under_root(project_root_path, config_file.string()));
        }
    }
    // Derive cv root from config path so examples/... and applications/... resolve from any cwd
    if (config_file.is_absolute()) {
        fs::path p = config_file.parent_path();
        for (int i = 0; i < 3 && p.has_parent_path(); ++i) p = p.parent_path();
        project_root_path = p;
    }

    YAML::Node app_cfg;
    try {
        if (!fs::exists(config_file)) {
            throw std::runtime_error("Config file not found: " + config_file.string());
        }
        app_cfg = YAML::LoadFile(config_file.string());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    if (use_camera) {
        if (app_cfg["camera_id"]) camera_id = app_cfg["camera_id"].as<int>();
    } else {
        if (video_path.empty()) {
            if (app_cfg["test_video"]) {
                video_path = resolve_under_root(project_root_path, app_cfg["test_video"].as<std::string>());
                std::cout << "Using test_video from " << config_file << ": " << video_path << std::endl;
            } else {
                std::cerr << "Error: --video not provided and test_video not found in " << config_file << std::endl;
                return -1;
            }
        } else {
            video_path = resolve_under_root(project_root_path, video_path);
        }
    }

    // Load labels (optional, from application yaml)
    std::vector<std::string> labels;
    try {
        if (app_cfg["label_file_path"]) {
            labels = vision_common::load_labels(resolve_under_root(project_root_path,
                app_cfg["label_file_path"].as<std::string>()));
        }
    } catch (...) {
        labels.clear();
    }
    if (!labels.empty()) {
        std::cout << "Loaded " << labels.size() << " labels from " << config_file << std::endl;
    }

    std::string tracker_config_path = app_cfg["tracker_config_path"]
        ? app_cfg["tracker_config_path"].as<std::string>() : "";
    std::string tracker_model_path = !tracker_model_path_cli.empty()
        ? tracker_model_path_cli
        : (app_cfg["tracker_model_path"] ? app_cfg["tracker_model_path"].as<std::string>() : "");
    if (tracker_config_path.empty()) {
        std::cerr << "Error: tracker_config_path is required in intrusion_detection.yaml" << std::endl;
        return -1;
    }
    const fs::path tracker_cfg_file = (project_root_path / tracker_config_path).lexically_normal();
    if (!fs::exists(tracker_cfg_file)) {
        std::cerr << "Error: Tracker config not found: " << tracker_cfg_file << std::endl;
        return -1;
    }
    tracker_model_path = resolve_under_root(project_root_path, tracker_model_path);

    std::unique_ptr<VisionService> service = VisionService::Create(
        tracker_cfg_file.string(),
        tracker_model_path,
        true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return -1;
    }

    try {
        cv::VideoCapture cap;
        if (use_camera) {
            cap.open(camera_id);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open camera: " << camera_id << std::endl;
                return -1;
            }
        } else {
            if (!fs::exists(video_path)) {
                std::cerr << "Error: Video file not found: " << video_path << std::endl;
                return -1;
            }
            cap.open(video_path);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open video: " << video_path << std::endl;
                return -1;
            }
        }

        cv::Mat first_frame;
        if (!cap.read(first_frame) || first_frame.empty()) {
            std::cerr << "Error: Could not read first frame" << std::endl;
            return -1;
        }

        const int frame_w = first_frame.cols;
        const int frame_h = first_frame.rows;

        std::vector<cv::Point> roi_pts = default_roi_polygon(frame_w, frame_h);
        std::cout << "Using default center ROI" << std::endl;

        std::cout << "Start processing... press 'q' to quit." << std::endl;
        int frame_idx = 0;
        cv::Mat frame = first_frame;
        auto t_prev = std::chrono::steady_clock::now();

        while (true) {
            if (frame.empty()) {
                if (!cap.read(frame) || frame.empty()) {
                    std::cout << "End of stream." << std::endl;
                    break;
                }
            }
            frame_idx++;
            auto t_now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_now - t_prev).count();
            double current_fps = (dt > 1e-6) ? 1.0 / dt : 0.0;
            t_prev = t_now;

            std::vector<VisionServiceResult> results;
            int ret = service->InferImage(frame, &results);
            if (ret != VISION_SERVICE_OK) {
                std::cerr << "Error: " << service->LastError() << std::endl;
                cap.release();
                cv::destroyAllWindows();
                return -1;
            }

        cv::Mat vis = draw_roi_overlay(frame, roi_pts, 0.22);
        std::set<int> inside_ids;

        for (const auto& r : results) {
            const int track_id = r.track_id;
            const int class_id = r.label;
            if (!is_person_label(class_id, labels)) {
                continue;
            }

            int x1 = std::clamp(static_cast<int>(r.x1), 0, frame_w - 1);
            int y1 = std::clamp(static_cast<int>(r.y1), 0, frame_h - 1);
            int x2 = std::clamp(static_cast<int>(r.x2), 0, frame_w - 1);
            int y2 = std::clamp(static_cast<int>(r.y2), 0, frame_h - 1);

            cv::Point foot((x1 + x2) / 2, y2);
            bool intrusion = point_in_roi(foot, roi_pts);
            if (intrusion && track_id >= 0) inside_ids.insert(track_id);

            cv::Scalar color = intrusion ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            int thickness = intrusion ? 3 : 2;

            cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
            cv::circle(vis, foot, 4, color, -1);

            std::string base_name = "obj";
            if (!labels.empty() && class_id >= 0 && class_id < static_cast<int>(labels.size())) {
                base_name = labels[class_id];
            }

            std::ostringstream oss;
            oss << base_name << " ID:" << track_id << " " << std::fixed << std::setprecision(2) << r.score;
            if (intrusion) oss << " INTRUSION";
            std::string text = oss.str();

            int baseline = 0;
            cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            int tx1 = x1;
            int ty1 = std::max(0, y1 - ts.height - baseline - 6);
            int tx2 = std::min(frame_w - 1, x1 + ts.width + 6);
            int ty2 = y1;
            cv::rectangle(vis, cv::Point(tx1, ty1), cv::Point(tx2, ty2), color, -1);
            cv::putText(vis, text, cv::Point(x1 + 3, y1 - 6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

            char fps_buf[32];
            std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", current_fps);
            cv::putText(vis, fps_buf, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(vis, "Frame: " + std::to_string(frame_idx), cv::Point(10, 65),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::putText(vis, "Intrusions (live): " + std::to_string(inside_ids.size()),
                        cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                        cv::Scalar(0, 0, 255), 2);

            cv::imshow("Intrusion Identification (ByteTrack)", vis);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                std::cout << "Quit." << std::endl;
                break;
            }

            frame.release();  // force next read
        }

        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

