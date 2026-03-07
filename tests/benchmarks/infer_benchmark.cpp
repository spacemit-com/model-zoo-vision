/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

// Expose protected BaseModel helpers for stage timing only.
#define protected public
#include "core/cpp/vision_model_factory.h"
#undef protected
#include "core/cpp/vision_task_interfaces.h"
#include "common/cpp/drawing.h"
#include "deploy/arcface/cpp/arcface_recognizer.h"
#include "deploy/emotion/cpp/emotion_recognizer.h"
#include "deploy/resnet/cpp/resnet_classifier.h"
#include "deploy/yolov5_face/cpp/yolov5_face_detector.h"
#include "deploy/yolov5_gesture/cpp/yolov5_gesture_detector.h"
#include "deploy/yolov8/cpp/yolov8_detector.h"
#include "deploy/yolov8_pose/cpp/yolov8_pose_detector.h"
#include "deploy/yolov8_seg/cpp/yolov8_seg_detector.h"

namespace {

struct Args {
    std::string config_path;
    std::string image_path;
    std::string model_path_override;
    int runs = 100;
    int warmup = 5;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
                << " --config <yaml> --image <path> [--model-path <path>] [--runs N] [--warmup N]\n"
                << "Example: " << prog
                << " --config examples/yolov8/config/yolov8.yaml --image assets/images/bus.jpg\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (a == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (a == "--model-path" && i + 1 < argc) {
            args.model_path_override = argv[++i];
        } else if (a == "--runs" && i + 1 < argc) {
            args.runs = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--warmup" && i + 1 < argc) {
            args.warmup = std::max(0, std::stoi(argv[++i]));
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if (args.config_path.empty() || args.image_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

double to_ms(const std::chrono::steady_clock::duration& d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// Run task via new task interfaces (replaces infer).
std::vector<vision_common::Result> run_task(vision_core::BaseModel* model, const cv::Mat& image) {
    using vision_core::IClassificationModel;
    using vision_core::IDetectionModel;
    using vision_core::IEmbeddingModel;
    using vision_core::IPoseModel;
    using vision_core::ISegmentationModel;
    using vision_core::ITrackingModel;
    using vision_core::ModelCapability;
    if (model->supports_capability(ModelCapability::kTrackUpdate)) {
        if (auto* p = dynamic_cast<ITrackingModel*>(model))
            return p->track(image);
    }
    if (auto* p = dynamic_cast<IPoseModel*>(model))
        return p->estimate_pose(image);
    if (auto* p = dynamic_cast<ISegmentationModel*>(model))
        return p->segment(image);
    if (auto* p = dynamic_cast<IDetectionModel*>(model))
        return p->detect(image);
    if (auto* p = dynamic_cast<IClassificationModel*>(model))
        return p->classify(image);
    if (auto* p = dynamic_cast<IEmbeddingModel*>(model)) {
        std::vector<float> emb = p->infer_embedding(image);
        if (emb.empty()) return {};
        vision_common::Result r;
        r.embedding = std::move(emb);
        r.score = 1.0f;
        r.label = -1;
        return {r};
    }
    return {};
}

// Call concrete preprocess for stage timing (BaseModel no longer has preprocess).
cv::Mat preprocess_blob(vision_core::BaseModel* model, const cv::Mat& image) {
    using vision_deploy::ArcFaceRecognizer;
    using vision_deploy::EmotionRecognizer;
    using vision_deploy::ResNetClassifier;
    using vision_deploy::YOLOv5FaceDetector;
    using vision_deploy::YOLOv5GestureDetector;
    using vision_deploy::YOLOv8Detector;
    using vision_deploy::YOLOv8PoseDetector;
    using vision_deploy::YOLOv8SegDetector;
    if (auto* p = dynamic_cast<YOLOv8Detector*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<YOLOv8PoseDetector*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<YOLOv8SegDetector*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<YOLOv5FaceDetector*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<YOLOv5GestureDetector*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<ResNetClassifier*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<EmotionRecognizer*>(model)) return p->preprocess(image);
    if (auto* p = dynamic_cast<ArcFaceRecognizer*>(model)) return p->preprocess(image);
    return cv::Mat();
}

// Call concrete postprocess (no common base method); used for postprocess timing only.
void call_postprocess(vision_core::BaseModel* model,
                    std::vector<Ort::Value>& outputs,
                    const cv::Size& orig_size) {
    using vision_deploy::ArcFaceRecognizer;
    using vision_deploy::EmotionRecognizer;
    using vision_deploy::ResNetClassifier;
    using vision_deploy::YOLOv5FaceDetector;
    using vision_deploy::YOLOv5GestureDetector;
    using vision_deploy::YOLOv8Detector;
    using vision_deploy::YOLOv8PoseDetector;
    using vision_deploy::YOLOv8SegDetector;
    if (auto* p = dynamic_cast<YOLOv8Detector*>(model)) {
        (void)p->postprocess(outputs, orig_size);
        return;
    }
    if (auto* p = dynamic_cast<YOLOv8PoseDetector*>(model)) {
        (void)p->postprocess(outputs, orig_size);
        return;
    }
    if (auto* p = dynamic_cast<YOLOv8SegDetector*>(model)) {
        (void)p->postprocess(outputs, orig_size);
        return;
    }
    if (auto* p = dynamic_cast<YOLOv5FaceDetector*>(model)) {
        (void)p->postprocess(outputs, orig_size);
        return;
    }
    if (auto* p = dynamic_cast<YOLOv5GestureDetector*>(model)) {
        (void)p->postprocess(outputs, orig_size);
        return;
    }
    if (auto* p = dynamic_cast<ResNetClassifier*>(model)) {
        (void)p->postprocess(outputs);
        return;
    }
    if (auto* p = dynamic_cast<EmotionRecognizer*>(model)) {
        (void)p->postprocess(outputs);
        return;
    }
    if (auto* p = dynamic_cast<ArcFaceRecognizer*>(model)) {
        (void)p->postprocess(outputs);
        return;
    }
    (void)outputs;
    (void)orig_size;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    if (!args.model_path_override.empty()) {
        args.model_path_override = vision_core::resolveResourcePath(
            args.model_path_override,
            args.config_path);
    }

    cv::Mat image = cv::imread(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image: " << args.image_path << std::endl;
        return 1;
    }

    std::unique_ptr<vision_core::BaseModel> model;
    try {
        model = vision_core::createModelFromConfigPath(
            args.config_path,
            args.model_path_override,
            true);
    } catch (const std::exception& e) {
        std::cerr << "Error: createModel failed: " << e.what() << std::endl;
        return 1;
    }

    // Warm-up (also ensures model/session is initialized).
    for (int i = 0; i < args.warmup; ++i) {
        try {
            (void)run_task(model.get(), image);
        } catch (const std::exception& e) {
            std::cerr << "Error: warmup run_task failed: " << e.what() << std::endl;
            return 1;
        }
    }
    if (args.warmup == 0) {
        // Ensure model is initialized before preprocess/run_session timings.
        model->ensure_model_loaded();
    }

    // Preprocess timing (concrete preprocess per model type).
    double preprocess_total = 0.0;
    cv::Mat preprocessed = preprocess_blob(model.get(), image);
    if (!preprocessed.empty()) {
        for (int i = 0; i < args.runs; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            cv::Mat blob = preprocess_blob(model.get(), image);
            const auto t1 = std::chrono::steady_clock::now();
            preprocess_total += to_ms(t1 - t0);
            (void)blob;
        }
    }

    double onnx_total = 0.0;
    const cv::Size orig_size = image.size();
    if (!preprocessed.empty()) {
        for (int i = 0; i < args.runs; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<Ort::Value> outputs = model->run_session(preprocessed);
            const auto t1 = std::chrono::steady_clock::now();
            onnx_total += to_ms(t1 - t0);
            (void)outputs;
        }
    }

    // Postprocess timing (run_session + postprocess per iteration; postprocess is timed only).
    double postprocess_total = 0.0;
    if (!preprocessed.empty()) {
        for (int i = 0; i < args.runs; ++i) {
            std::vector<Ort::Value> outputs = model->run_session(preprocessed);
            const auto t0 = std::chrono::steady_clock::now();
            call_postprocess(model.get(), outputs, orig_size);
            const auto t1 = std::chrono::steady_clock::now();
            postprocess_total += to_ms(t1 - t0);
        }
    }

    // Full task timing (detect/classify/track/estimate_pose/segment/infer_embedding).
    double infer_total = 0.0;
    std::vector<vision_common::Result> last_results;
    for (int i = 0; i < args.runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_results = run_task(model.get(), image);
        const auto t1 = std::chrono::steady_clock::now();
        infer_total += to_ms(t1 - t0);
    }

    // Draw timing (same logic as vision_service_draw: pick draw_* by result type).
    double draw_total = 0.0;
    if (!last_results.empty()) {
        bool has_keypoints = false;
        bool has_mask = false;
        bool has_track_id = false;
        for (const auto& r : last_results) {
            if (!r.keypoints.empty()) has_keypoints = true;
            if (r.mask != nullptr && !r.mask->empty()) has_mask = true;
            if (r.track_id >= 0) has_track_id = true;
        }
        std::vector<std::string> labels;  // empty for benchmark
        for (int i = 0; i < args.runs; ++i) {
            cv::Mat vis = image.clone();
            const auto t0 = std::chrono::steady_clock::now();
            if (has_keypoints) {
                vision_common::draw_keypoints(vis, last_results);
            } else if (has_mask) {
                vision_common::draw_segmentation(vis, last_results, labels);
            } else if (has_track_id) {
                vision_common::draw_tracking_results(vis, last_results, labels);
            } else {
                vision_common::draw_detections(vis, last_results, labels);
            }
            const auto t1 = std::chrono::steady_clock::now();
            draw_total += to_ms(t1 - t0);
        }
    }

    const double avg_pre = preprocess_total / args.runs;
    const double avg_onnx = onnx_total / args.runs;
    const double avg_post = postprocess_total / args.runs;
    const double avg_infer = infer_total / args.runs;
    const double avg_draw = (args.runs > 0 && !last_results.empty())
                                ? (draw_total / args.runs)
                                : 0.0;

    const double fps = (avg_infer > 0.0) ? (1000.0 / avg_infer) : 0.0;

    std::cout << "Config: " << args.config_path << "\n"
                << "Image: " << args.image_path << "\n"
                << "Runs: " << args.runs << " (warmup " << args.warmup << ")\n"
                << "Avg preprocess: " << avg_pre << " ms\n"
                << "Avg ONNX Runtime: " << avg_onnx << " ms\n"
                << "Avg postprocess (measured): " << avg_post << " ms\n"
                << "Avg draw: " << avg_draw << " ms\n"
                << "Avg infer (total, no draw): " << avg_infer << " ms\n"
                << "FPS (avg infer): " << fps << "\n";

    return 0;
}
