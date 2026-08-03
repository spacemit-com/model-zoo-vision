/*
* Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
* SPDX-License-Identifier: Apache-2.0
*/

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>  // OpenCV 5: invertAffineTransform
#endif
#include <yaml-cpp/yaml.h>

#include "align_face.h"
#include "face_recognition_runtime.h"
#include "vision_service.h"

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultAppConfig =
    "applications/face_recognition/config/face_recognition.yaml";
constexpr const char* kDefaultOutputImage = "output_face_recognition.jpg";

struct FaceDrawInfo {
    std::string label;
    cv::Scalar box_color{cv::Scalar(0, 255, 0)};
    std::vector<vision::KeyPoint> extra_landmarks;
};


bool LooksLikeYamlPath(const std::string& path) {
    if (path.size() < 5) {
        return false;
    }
    return path.compare(path.size() - 5, 5, ".yaml") == 0 ||
        path.compare(path.size() - 4, 4, ".yml") == 0;
}

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
    const fs::path cwd = fs::current_path();
    if (IsRepoRoot(cwd)) {
        return fs::absolute(cwd);
    }
    if (cwd.filename() == "build" && cwd.has_parent_path()) {
        const fs::path parent = cwd.parent_path();
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
    const fs::path candidates[] = {project_root / in, fs::current_path() / in};
    for (const fs::path& candidate : candidates) {
        const fs::path abs = fs::absolute(candidate).lexically_normal();
        if (fs::exists(abs)) {
            return abs.string();
        }
    }
    return fs::absolute(project_root / in).lexically_normal().string();
}

std::string ResolveConfigPath(const fs::path& config_dir,
                                const fs::path& project_root,
                                const std::string& path) {
    if (path.empty()) {
        return "";
    }
    const fs::path local = config_dir / path;
    if (fs::exists(local)) {
        return local.lexically_normal().string();
    }
    return ResolveUserPath(project_root, path);
}

std::string YamlString(const YAML::Node& node, const char* key) {
    if (!node[key]) {
        return "";
    }
    return node[key].as<std::string>();
}

std::unique_ptr<VisionService> CreateModelService(const fs::path& config_dir,
                                                    const fs::path& project_root,
                                                    const std::string& rel_path) {
    const std::string abs = ResolveConfigPath(config_dir, project_root, rel_path);
    if (abs.empty()) {
        return nullptr;
    }
    return VisionService::Create(abs, "", false);
}

VisionServiceStatus InferPoseList(VisionService* service,
                                    const cv::Mat& image,
                                    std::vector<vision::Pose>* out) {
    out->clear();
    VisionServiceResponse response;
    const VisionServiceStatus ret = service->Infer(image, &response);
    if (ret != VISION_SERVICE_OK) {
        return ret;
    }
    for (const auto& item : response.results) {
        if (const vision::Pose* pose = std::get_if<vision::Pose>(&item)) {
            out->push_back(*pose);
        }
    }
    return out->empty() ? VISION_SERVICE_INFER_FAILED : VISION_SERVICE_OK;
}

VisionServiceStatus InferEmbedding(VisionService* service,
                                    const cv::Mat& image,
                                    std::vector<float>* out) {
    out->clear();
    VisionServiceResponse response;
    const VisionServiceStatus ret = service->Infer(image, &response);
    if (ret != VISION_SERVICE_OK) {
        return ret;
    }
    if (response.results.empty()) {
        return VISION_SERVICE_INFER_FAILED;
    }
    const vision::Embedding* emb = std::get_if<vision::Embedding>(&response.results[0]);
    if (emb == nullptr) {
        return VISION_SERVICE_INFER_FAILED;
    }
    *out = emb->embedding;
    return VISION_SERVICE_OK;
}

VisionServiceStatus InferClassification(VisionService* service,
                                        const cv::Mat& image,
                                        vision::Classification* out) {
    VisionServiceResponse response;
    const VisionServiceStatus ret = service->Infer(image, &response);
    if (ret != VISION_SERVICE_OK) {
        return ret;
    }
    if (response.results.empty()) {
        return VISION_SERVICE_INFER_FAILED;
    }
    const vision::Classification* cls = std::get_if<vision::Classification>(&response.results[0]);
    if (cls == nullptr) {
        return VISION_SERVICE_INFER_FAILED;
    }
    *out = *cls;
    return VISION_SERVICE_OK;
}

cv::Mat CropFace(const cv::Mat& image, const vision::BoundingBox& bbox) {
    const int x = std::max(0, static_cast<int>(std::floor(bbox.x1)));
    const int y = std::max(0, static_cast<int>(std::floor(bbox.y1)));
    const int x2 = std::min(image.cols, static_cast<int>(std::ceil(bbox.x2)));
    const int y2 = std::min(image.rows, static_cast<int>(std::ceil(bbox.y2)));
    const int w = std::max(0, x2 - x);
    const int h = std::max(0, y2 - y);
    if (w <= 0 || h <= 0) {
        return {};
    }
    return image(cv::Rect(x, y, w, h)).clone();
}

bool BuildLandmark106Input(const cv::Mat& image,
                            const vision::BoundingBox& bbox,
                            int input_size,
                            cv::Mat* out_crop,
                            cv::Mat* out_inv_affine) {
    if (image.empty() || out_crop == nullptr || out_inv_affine == nullptr || input_size <= 0) {
        return false;
    }
    const float w = bbox.x2 - bbox.x1;
    const float h = bbox.y2 - bbox.y1;
    if (w <= 1.0f || h <= 1.0f) {
        return false;
    }

    // Match InsightFace landmark crop policy:
    // center = bbox center, scale = input_size / (max(w, h) * 1.5), rotate = 0.
    const float cx = (bbox.x1 + bbox.x2) * 0.5f;
    const float cy = (bbox.y1 + bbox.y2) * 0.5f;
    const float side = std::max(w, h) * 1.5f;
    if (side <= 1e-6f) {
        return false;
    }
    const float s = static_cast<float>(input_size) / side;

    cv::Mat_<double> affine(2, 3);
    affine(0, 0) = static_cast<double>(s);
    affine(0, 1) = 0.0;
    affine(0, 2) = static_cast<double>(input_size * 0.5f - cx * s);
    affine(1, 0) = 0.0;
    affine(1, 1) = static_cast<double>(s);
    affine(1, 2) = static_cast<double>(input_size * 0.5f - cy * s);

    cv::warpAffine(image, *out_crop, affine, cv::Size(input_size, input_size),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::invertAffineTransform(affine, *out_inv_affine);
    return !out_crop->empty();
}

cv::Mat AlignFromPose(const cv::Mat& image, const vision::Pose& pose) {
    if (pose.keypoints.size() < 5) {
        return {};
    }
    cv::Point2f landmarks[5];
    for (int i = 0; i < 5; ++i) {
        landmarks[i] = cv::Point2f(pose.keypoints[static_cast<size_t>(i)].x,
                                    pose.keypoints[static_cast<size_t>(i)].y);
    }
    return vision_common::align_face_5pt(image, landmarks, 112);
}

bool SaveEmbedding(const fs::path& path, const std::vector<float>& embedding) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    const uint64_t dim = embedding.size();
    file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    file.write(reinterpret_cast<const char*>(embedding.data()),
                static_cast<std::streamsize>(embedding.size() * sizeof(float)));
    return static_cast<bool>(file);
}

bool LoadEmbedding(const fs::path& path, std::vector<float>* embedding) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    uint64_t dim = 0;
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (dim == 0 || dim > 100000) {
        return false;
    }
    embedding->resize(static_cast<size_t>(dim));
    file.read(reinterpret_cast<char*>(embedding->data()),
                static_cast<std::streamsize>(dim * sizeof(float)));
    return static_cast<bool>(file);
}

void DrawFaceAnnotation(cv::Mat& canvas,
                        const vision::Pose& face,
                        const FaceDrawInfo& info) {
    const int x1 = static_cast<int>(face.bbox.x1);
    const int y1 = static_cast<int>(face.bbox.y1);
    const int x2 = static_cast<int>(face.bbox.x2);
    const int y2 = static_cast<int>(face.bbox.y2);
    cv::rectangle(canvas, cv::Point(x1, y1), cv::Point(x2, y2), info.box_color, 2);

    const cv::Scalar kp_color(0, 0, 255);
    const size_t num_kp = std::min(face.keypoints.size(), static_cast<size_t>(5));
    for (size_t i = 0; i < num_kp; ++i) {
        const auto& kp = face.keypoints[i];
        const cv::Point pt(static_cast<int>(kp.x), static_cast<int>(kp.y));
        cv::circle(canvas, pt, 3, kp_color, -1);
    }

    for (const auto& kp : info.extra_landmarks) {
        const cv::Point pt(static_cast<int>(kp.x), static_cast<int>(kp.y));
        // Use high-contrast cyan markers so 106 landmarks are unmistakable.
        cv::circle(canvas, pt, 3, cv::Scalar(255, 255, 0), -1);
    }

    int text_y = std::max(20, y1 - 8);
    cv::putText(canvas, info.label, cv::Point(x1, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, info.box_color, 2);
    if (!info.extra_landmarks.empty()) {
        cv::putText(canvas,
                    "lm106=" + std::to_string(info.extra_landmarks.size()),
                    cv::Point(x1, std::min(y2 + 20, canvas.rows - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
}

bool SaveResultImage(const cv::Mat& image,
                    const std::vector<vision::Pose>& faces,
                    const std::vector<FaceDrawInfo>& infos,
                    const std::string& output_path) {
    if (image.empty() || faces.size() != infos.size()) {
        return false;
    }
    cv::Mat canvas = image.clone();
    for (size_t i = 0; i < faces.size(); ++i) {
        DrawFaceAnnotation(canvas, faces[i], infos[i]);
    }
    return cv::imwrite(output_path, canvas);
}

bool ProcessImagePipeline(
    const cv::Mat& image,
    VisionService* scrfd,
    VisionService* arcface,
    VisionService* genderage,
    VisionService* landmark106,
    bool enable_landmark106,
    bool enable_recognition,
    bool run_embedding,
    bool verbose,
    const std::string& face_db_dir,
    float recognize_threshold,
    std::vector<vision::Pose>* faces_out,
    std::vector<FaceDrawInfo>* draw_infos_out) {
    faces_out->clear();
    draw_infos_out->clear();
    if (InferPoseList(scrfd, image, faces_out) != VISION_SERVICE_OK) {
        const std::string det_err = scrfd->LastError();
        // Some backends report "no detection" via non-OK status with empty error.
        if (!det_err.empty()) {
            std::cerr << "Error: face detection failed: " << det_err << std::endl;
        } else if (verbose) {
            std::cerr << "No face detected" << std::endl;
        }
        return false;
    }
    if (faces_out->empty()) {
        std::cerr << "No face detected" << std::endl;
        return false;
    }
    std::sort(faces_out->begin(), faces_out->end(),
                [](const vision::Pose& a, const vision::Pose& b) { return a.score > b.score; });

    draw_infos_out->assign(faces_out->size(), FaceDrawInfo{});
    for (size_t fi = 0; fi < faces_out->size(); ++fi) {
        const cv::Mat aligned = AlignFromPose(image, (*faces_out)[fi]);
        if (aligned.empty()) {
            (*draw_infos_out)[fi].label = "#" + std::to_string(fi) + " align:failed";
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 165, 255);
            continue;
        }

        std::vector<float> query;
        const bool has_embedding = run_embedding
            ? (InferEmbedding(arcface, aligned, &query) == VISION_SERVICE_OK && !query.empty())
            : false;

        vision::Classification ga{};
        const bool has_ga = InferClassification(genderage, aligned, &ga) == VISION_SERVICE_OK &&
                            ga.class_scores.size() >= 3;
        if (has_ga && verbose) {
            const char* gender = (ga.label == 1) ? "male" : "female";
            std::cout << "Face " << fi << " gender/age: " << gender << ", age="
                        << static_cast<int>(ga.class_scores[2]) << std::endl;
        }

        if (enable_landmark106 && landmark106 != nullptr) {
            cv::Mat lm_crop;
            cv::Mat lm_inv_affine;
            if (BuildLandmark106Input(image, (*faces_out)[fi].bbox, 192, &lm_crop, &lm_inv_affine)) {
                std::vector<vision::Pose> lm_faces;
                if (InferPoseList(landmark106, lm_crop, &lm_faces) == VISION_SERVICE_OK &&
                    !lm_faces.empty()) {
                    if (verbose) {
                        std::cout << "Face " << fi << " landmark106 points: "
                                    << lm_faces[0].keypoints.size() << std::endl;
                    }
                    (*draw_infos_out)[fi].extra_landmarks.reserve(lm_faces[0].keypoints.size());
                    for (const auto& kp : lm_faces[0].keypoints) {
                        vision::KeyPoint mapped = kp;
                        const double x = static_cast<double>(kp.x);
                        const double y = static_cast<double>(kp.y);
                        mapped.x = static_cast<float>(
                            lm_inv_affine.at<double>(0, 0) * x +
                            lm_inv_affine.at<double>(0, 1) * y +
                            lm_inv_affine.at<double>(0, 2));
                        mapped.y = static_cast<float>(
                            lm_inv_affine.at<double>(1, 0) * x +
                            lm_inv_affine.at<double>(1, 1) * y +
                            lm_inv_affine.at<double>(1, 2));
                        (*draw_infos_out)[fi].extra_landmarks.push_back(mapped);
                    }
                }
            }
        }

        std::ostringstream label;
        label << "#" << fi << " det:" << std::fixed << std::setprecision(2) << (*faces_out)[fi].score;
        if (has_ga) {
            label << " " << ((ga.label == 1) ? "M" : "F")
                    << " age:" << static_cast<int>(ga.class_scores[2]);
        }
        if (run_embedding && !has_embedding) {
            label << " emb:failed";
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 165, 255);
            (*draw_infos_out)[fi].label = label.str();
            continue;
        }

        if (!enable_recognition || !run_embedding) {
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 255, 0);
            (*draw_infos_out)[fi].label = label.str();
            continue;
        }

        std::string best_name;
        float best_score = -1.0f;
        if (verbose) {
            std::cout << "\nSimilarity scores (face " << fi << "):" << std::endl;
        }
        if (!fs::exists(face_db_dir)) {
            if (verbose) {
                std::cout << "  (empty face db: " << face_db_dir << ")" << std::endl;
            }
        } else {
            for (const auto& entry : fs::directory_iterator(face_db_dir)) {
                if (!entry.is_regular_file() || entry.path().extension() != ".bin") {
                    continue;
                }
                std::vector<float> stored;
                if (!LoadEmbedding(entry.path(), &stored)) {
                    continue;
                }
                if (stored.size() != query.size()) {
                    continue;
                }
                const float score = VisionService::EmbeddingSimilarity(query, stored);
                const std::string db_name = entry.path().stem().string();
                if (verbose) {
                    std::cout << "  " << db_name << ": " << std::fixed << std::setprecision(4) << score
                                << std::endl;
                }
                if (score > best_score) {
                    best_score = score;
                    best_name = db_name;
                }
            }
        }

        const bool known = best_score >= recognize_threshold;
        if (known) {
            label << " " << best_name << ":" << std::setprecision(2) << best_score;
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 255, 0);
        } else if (best_score >= 0.0f) {
            label << " Unknown:" << std::setprecision(2) << best_score;
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 165, 255);
        } else {
            (*draw_infos_out)[fi].box_color = cv::Scalar(0, 165, 255);
        }
        (*draw_infos_out)[fi].label = label.str();
        if (verbose) {
            std::cout << "Best match (face " << fi << "): ";
            if (known) {
                std::cout << best_name << " (score: " << best_score << ")" << std::endl;
            } else {
                std::cout << "Unknown person (best: " << best_score << ")" << std::endl;
            }
        }
    }
    return true;
}

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog
                << " [--image <path>] [output_path]\n"
                << "      [--use-camera] [--camera-id <id>] [--camera-width <w>] [--camera-height <h>] [--camera-skip <n>]\n"
                << "      [--register <name>] [--recognize] [--save-image|--no-save-image]\n"
                << "      [--config <app.yaml>]  (optional; default: applications/face_recognition/config/face_recognition.yaml)\n"
                << "\nDefaults:\n"
                << "  (none): analyze with config test_image, save image enabled\n"
                << "  --register <name>: enroll face from --image / positional image\n"
                << "  --recognize: match against face db\n"
                << "  --use-camera: live camera; add --recognize to enable matching\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    const fs::path project_root = FindProjectRoot((argc > 0 && argv[0]) ? fs::path(argv[0]) : fs::path());
    std::string app_config_rel = kDefaultAppConfig;
    std::string image_path;
    std::string output_path_arg;
    std::string register_name;
    bool do_recognize = false;
    bool save_image_override = false;
    bool save_image_value = false;
    bool use_camera_flag = false;
    int camera_id_flag = 0;
    int camera_width = 0;
    int camera_height = 0;
    int camera_skip = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            app_config_rel = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path_arg = argv[++i];
        } else if (arg == "--register" && i + 1 < argc) {
            register_name = argv[++i];
        } else if (arg == "--recognize") {
            do_recognize = true;
        } else if (arg == "--save-image") {
            save_image_override = true;
            save_image_value = true;
        } else if (arg == "--no-save-image") {
            save_image_override = true;
            save_image_value = false;
        } else if (arg == "--use-camera") {
            use_camera_flag = true;
        } else if (arg == "--camera-id" && i + 1 < argc) {
            try {
                camera_id_flag = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Error: invalid --camera-id value" << std::endl;
                return 1;
            }
        } else if (arg == "--camera-width" && i + 1 < argc) {
            camera_width = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--camera-height" && i + 1 < argc) {
            camera_height = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--camera-skip" && i + 1 < argc) {
            camera_skip = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            if (LooksLikeYamlPath(arg) && app_config_rel == kDefaultAppConfig) {
                app_config_rel = arg;
            } else if (image_path.empty()) {
                image_path = arg;
            } else if (output_path_arg.empty()) {
                output_path_arg = arg;
            } else {
                std::cerr << "Error: unexpected argument: " << arg << std::endl;
                PrintUsage(argv[0]);
                return 1;
            }
        } else {
            std::cerr << "Error: unknown option: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (!register_name.empty() && (do_recognize || use_camera_flag)) {
        std::cerr << "Error: --register cannot be combined with --recognize or --use-camera"
            << std::endl;
        return 1;
    }

    const bool is_register = !register_name.empty();
    const bool is_camera = use_camera_flag;
    const bool is_recognize = do_recognize && !is_camera;
    // camera + --recognize: live matching; bare --recognize: image matching
    const bool enable_recognition = do_recognize;

    const fs::path app_config_path(ResolveUserPath(project_root, app_config_rel));
    if (!fs::exists(app_config_path)) {
        std::cerr << "Error: app config not found: " << app_config_path << std::endl;
        std::cerr << "Hint: run from repo root or pass --config "
                    << "applications/face_recognition/config/face_recognition.yaml" << std::endl;
        return 1;
    }

    YAML::Node app_cfg;
    try {
        app_cfg = YAML::LoadFile(app_config_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to parse app config: " << e.what() << std::endl;
        return 1;
    }

    const fs::path config_dir = app_config_path.parent_path();
    const float recognize_threshold = app_cfg["recognize_threshold"]
        ? app_cfg["recognize_threshold"].as<float>()
        : 0.3f;
    const bool enable_landmark106 = app_cfg["enable_landmark106"]
        ? app_cfg["enable_landmark106"].as<bool>()
        : false;
    const std::string face_db_dir = ResolveUserPath(
        project_root, YamlString(app_cfg, "face_db_dir").empty()
                            ? std::string("~/.cache/face_db")
                            : YamlString(app_cfg, "face_db_dir"));
    std::string output_image = output_path_arg;
    if (output_image.empty()) {
        output_image = YamlString(app_cfg, "output_image");
    }
    if (output_image.empty()) {
        output_image = kDefaultOutputImage;
    }
    output_image = ExpandTilde(output_image);

    const bool save_image_default = !is_register && !is_recognize && !is_camera;
    const bool save_image = save_image_override ? save_image_value : save_image_default;

    auto scrfd = CreateModelService(config_dir, project_root, YamlString(app_cfg, "scrfd_model"));
    auto arcface = CreateModelService(config_dir, project_root, YamlString(app_cfg, "arcface_model"));
    auto genderage = CreateModelService(config_dir, project_root, YamlString(app_cfg, "genderage_model"));
    std::unique_ptr<VisionService> landmark106;
    if (enable_landmark106) {
        landmark106 = CreateModelService(config_dir, project_root, YamlString(app_cfg, "landmark106_model"));
    }

    if (!scrfd || !arcface || !genderage) {
        std::cerr << "Error: failed to create model services. "
                    << VisionService::LastCreateError() << std::endl;
        return 1;
    }
    if (enable_landmark106 && !landmark106) {
        std::cerr << "Error: enable_landmark106=true but landmark service failed." << std::endl;
        return 1;
    }

    if (is_register) {
        if (image_path.empty()) {
            std::cerr << "Error: --register requires --image or a positional image path" << std::endl;
            return 1;
        }
        const std::string name = register_name;
        const std::string resolved_image = ResolveUserPath(project_root, image_path);
        const cv::Mat image = cv::imread(resolved_image);
        if (image.empty()) {
            std::cerr << "Error: failed to load image: " << resolved_image << std::endl;
            return 1;
        }

        std::vector<vision::Pose> faces;
        std::vector<FaceDrawInfo> draw_infos;
        if (!ProcessImagePipeline(image, scrfd.get(), arcface.get(), genderage.get(), landmark106.get(),
                                    enable_landmark106, false,
                                    face_recognition::RunEmbeddingInRegistrationPipeline(), true,
                                    face_db_dir, recognize_threshold,
                                    &faces, &draw_infos)) {
            return 1;
        }

        const cv::Mat aligned = AlignFromPose(image, faces.front());
        if (aligned.empty()) {
            std::cerr << "Error: face alignment failed" << std::endl;
            return 1;
        }
        std::vector<float> embedding;
        if (InferEmbedding(arcface.get(), aligned, &embedding) != VISION_SERVICE_OK) {
            std::cerr << "Error: embedding failed: " << arcface->LastError() << std::endl;
            return 1;
        }

        fs::create_directories(face_db_dir);
        const fs::path out_path = fs::path(face_db_dir) / (name + ".bin");
        if (!SaveEmbedding(out_path, embedding)) {
            std::cerr << "Error: failed to save embedding to " << out_path << std::endl;
            return 1;
        }
        std::cout << "Registered: " << name << " -> " << out_path << std::endl;

        draw_infos.front().label += " register:" + name;
        if (save_image) {
            if (SaveResultImage(image, faces, draw_infos, output_image)) {
                std::cout << "Saved result image: " << output_image << std::endl;
            } else {
                std::cerr << "Warning: failed to save result image: " << output_image << std::endl;
            }
        }
        return 0;
    }

    if (is_camera) {
        cv::VideoCapture cap(camera_id_flag);
        if (!cap.isOpened()) {
            std::cerr << "Error: failed to open camera index " << camera_id_flag << std::endl;
            return 1;
        }
        if (camera_width > 0) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(camera_width));
        }
        if (camera_height > 0) {
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(camera_height));
        }
        int frame_idx = 0;
        constexpr int kCameraReadFailureLimit = 30;
        constexpr auto kCameraReadFailureBackoff = std::chrono::milliseconds(100);
        face_recognition::CameraReadFailurePolicy read_failure_policy(kCameraReadFailureLimit);
        std::cout << "Camera started. Press q or ESC to quit." << std::endl;
        while (true) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                if (!read_failure_policy.OnReadFailure()) {
                    std::cerr << "Error: camera read failed "
                                << read_failure_policy.consecutive_failures()
                                << " consecutive times; stopping." << std::endl;
                    return 1;
                }
                std::this_thread::sleep_for(kCameraReadFailureBackoff);
                continue;
            }
            read_failure_policy.OnReadSuccess();
            ++frame_idx;
            if (camera_skip > 0 && (frame_idx % (camera_skip + 1)) != 1) {
                cv::imshow("face_recognition_camera", frame);
                const int key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 27) {
                    break;
                }
                continue;
            }
            std::vector<vision::Pose> faces;
            std::vector<FaceDrawInfo> infos;
            const bool ok = ProcessImagePipeline(frame, scrfd.get(), arcface.get(), genderage.get(), landmark106.get(),
                                                enable_landmark106, enable_recognition,
                                                enable_recognition, false, face_db_dir,
                                                recognize_threshold, &faces, &infos);
            if (ok) {
                for (size_t i = 0; i < faces.size(); ++i) {
                    DrawFaceAnnotation(frame, faces[i], infos[i]);
                }
            }
            cv::imshow("face_recognition_camera", frame);
            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                break;
            }
        }
        cv::destroyAllWindows();
        return 0;
    }

    // analyze / recognize (image)
    if (image_path.empty() && app_cfg["test_image"]) {
        image_path = app_cfg["test_image"].as<std::string>();
    }
    if (image_path.empty()) {
        std::cerr << "Error: provide --image / positional image, or set test_image in config"
                    << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string resolved_image = ResolveUserPath(project_root, image_path);
    const cv::Mat image = cv::imread(resolved_image);
    if (image.empty()) {
        std::cerr << "Error: failed to load image: " << resolved_image << std::endl;
        return 1;
    }
    std::vector<vision::Pose> faces;
    std::vector<FaceDrawInfo> draw_infos;
    if (!ProcessImagePipeline(image, scrfd.get(), arcface.get(), genderage.get(), landmark106.get(),
                                enable_landmark106, enable_recognition, true, true, face_db_dir,
                                recognize_threshold, &faces, &draw_infos)) {
        return 1;
    }
    if (save_image) {
        if (SaveResultImage(image, faces, draw_infos, output_image)) {
            std::cout << "Saved result image: " << output_image << std::endl;
        } else {
            std::cerr << "Warning: failed to save result image: " << output_image << std::endl;
        }
    }
    return 0;
}
