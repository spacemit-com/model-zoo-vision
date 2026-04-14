/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_service.h"

namespace {

struct Args {
    std::string config_path;
    std::string image_path;
    std::string image_path2;
    std::string model_path_override;
    int runs = 100;
    int warmup = 5;
    bool verbose_timing = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
        << " --config <yaml> [--image <path>] [--model-path <path>] [--runs N] "
        << "[--image2 <path>] [--warmup N] [--verbose-timing]\n"
        << "Example: " << prog
        << " --config examples/yolov8/config/yolov8.yaml\n"
        << "Note: --config is required. If --image is omitted, benchmark falls back to "
        << "test_image/test_image1/test_video from config.\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            args.config_path = argv[++i];
        } else if (a == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (a == "--image2" && i + 1 < argc) {
            args.image_path2 = argv[++i];
        } else if (a == "--model-path" && i + 1 < argc) {
            args.model_path_override = argv[++i];
        } else if (a == "--runs" && i + 1 < argc) {
            args.runs = std::max(1, std::stoi(argv[++i]));
        } else if (a == "--warmup" && i + 1 < argc) {
            args.warmup = std::max(0, std::stoi(argv[++i]));
        } else if (a == "--verbose-timing") {
            args.verbose_timing = true;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if (args.config_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

double to_ms(const std::chrono::steady_clock::duration& d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

std::string pick_default_image(VisionService* service) {
    if (service == nullptr) {
        return {};
    }
    std::string image = service->GetDefaultImage();
    if (!image.empty()) {
        return image;
    }
    image = service->GetConfigPathValue("test_image1");
    if (!image.empty()) {
        return image;
    }
    std::string video_path = service->GetConfigPathValue("test_video");
    if (!video_path.empty()) {
        return video_path;
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    auto service = VisionService::Create(args.config_path, args.model_path_override, true);
    if (!service) {
        std::cerr << "Error: create service failed: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    VisionServiceTimingOptions timing_options;
    timing_options.enabled = true;
    timing_options.print_to_stdout = args.verbose_timing;
    service->SetTimingOptions(timing_options);

    if (args.image_path.empty()) {
        args.image_path = pick_default_image(service.get());
        if (args.image_path.empty()) {
            std::cerr << "Error: --image is not set and config has no valid "
                "test_image/test_image1/test_video"
                << std::endl;
            return 1;
        }
    }
    if (args.image_path2.empty()) {
        args.image_path2 = service->GetConfigPathValue("test_image2");
    }

    cv::Mat image;
    const std::string test_video_from_config = service->GetConfigPathValue("test_video");
    if (!test_video_from_config.empty() && args.image_path == test_video_from_config) {
        cv::VideoCapture cap(args.image_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video: " << args.image_path << std::endl;
            return 1;
        }
        cap >> image;
        if (image.empty()) {
            std::cerr << "Error: Could not read first frame from video: " << args.image_path << std::endl;
            return 1;
        }
    } else {
        image = cv::imread(args.image_path);
    }
    if (image.empty()) {
        std::cerr << "Error: Could not read image: " << args.image_path << std::endl;
        return 1;
    }

    std::vector<VisionServiceResult> results;
    std::vector<float> embedding;
    enum class BenchMode {
        kImageInfer,
        kEmbeddingInfer
    };
    BenchMode mode = BenchMode::kImageInfer;

    {
        VisionServiceStatus probe = service->InferImage(image, &results);
        if (probe != VISION_SERVICE_OK) {
            probe = service->InferEmbedding(image, &embedding);
            if (probe != VISION_SERVICE_OK) {
                std::cerr << "Error: neither InferImage nor InferEmbedding is supported for this config. "
                    << "Last error: " << service->LastError() << std::endl;
                return 1;
            }
            mode = BenchMode::kEmbeddingInfer;
        }
    }

    for (int i = 0; i < args.warmup; ++i) {
        VisionServiceStatus st = VISION_SERVICE_OK;
        if (mode == BenchMode::kImageInfer) {
            st = service->InferImage(image, &results);
            if (st != VISION_SERVICE_OK) {
                std::cerr << "Error: warmup InferImage failed: " << service->LastError() << std::endl;
                return 1;
            }
        } else {
            st = service->InferEmbedding(image, &embedding);
            if (st != VISION_SERVICE_OK) {
                std::cerr << "Error: warmup InferEmbedding failed: " << service->LastError() << std::endl;
                return 1;
            }
        }
    }

    double infer_total = 0.0;
    double draw_total = 0.0;
    double service_embedding_total = 0.0;
    double service_preprocess_total = 0.0;
    double service_model_infer_total = 0.0;
    double service_postprocess_total = 0.0;
    double service_detect_total = 0.0;
    double service_track_total = 0.0;
    int draw_skipped_runs = 0;
    int draw_executed_runs = 0;
    const bool draw_supported = service->SupportsDraw();

    cv::Mat drawn;

    for (int i = 0; i < args.runs; ++i) {
        VisionServiceStatus st = VISION_SERVICE_OK;
        const auto t0 = std::chrono::steady_clock::now();
        if (mode == BenchMode::kImageInfer) {
            st = service->InferImage(image, &results);
        } else {
            st = service->InferEmbedding(image, &embedding);
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (st != VISION_SERVICE_OK) {
            std::cerr << "Error: infer failed: " << service->LastError() << std::endl;
            return 1;
        }

        infer_total += to_ms(t1 - t0);
        VisionServiceTiming timing_after_infer = service->GetLastTiming();
        service_embedding_total += timing_after_infer.embedding_ms;
        service_preprocess_total += timing_after_infer.preprocess_ms;
        service_model_infer_total += timing_after_infer.model_infer_ms;
        service_postprocess_total += timing_after_infer.postprocess_ms;
        service_detect_total += timing_after_infer.detect_ms;
        service_track_total += timing_after_infer.track_ms;

        if (mode == BenchMode::kEmbeddingInfer) {
            ++draw_skipped_runs;
            continue;
        }

        bool skip_draw = results.empty() || !draw_supported;
        if (skip_draw) {
            ++draw_skipped_runs;
        } else {
            const auto d0 = std::chrono::steady_clock::now();
            st = service->Draw(image, &drawn);
            const auto d1 = std::chrono::steady_clock::now();

            if (st != VISION_SERVICE_OK) {
                std::cerr << "Error: Draw failed: " << service->LastError() << std::endl;
                return 1;
            }
            ++draw_executed_runs;
            draw_total += to_ms(d1 - d0);
        }
    }

    const double avg_infer = infer_total / args.runs;
    const double avg_service_embedding = service_embedding_total / args.runs;
    const double avg_service_preprocess = service_preprocess_total / args.runs;
    const double avg_service_model_infer = service_model_infer_total / args.runs;
    const double avg_service_postprocess = service_postprocess_total / args.runs;
    const double avg_service_detect = service_detect_total / args.runs;
    const double avg_service_track = service_track_total / args.runs;
    const double avg_draw = (draw_executed_runs > 0) ? (draw_total / draw_executed_runs) : 0.0;
    const double fps = (avg_infer > 0.0) ? (1000.0 / avg_infer) : 0.0;
    const bool is_embedding_mode = (mode == BenchMode::kEmbeddingInfer);
    const bool looks_like_tracking = !is_embedding_mode &&
        ((avg_service_detect > 0.0) || (avg_service_track > 0.0));

    std::string mode_name = (mode == BenchMode::kEmbeddingInfer) ? "InferEmbedding" : "InferImage";
    std::cout << "Mode: " << mode_name << "\n"
        << "Config: " << args.config_path << "\n"
        << "Image: " << args.image_path << "\n"
        << "Runs: " << args.runs << ", warmup " << args.warmup << "\n"
        << "Avg infer: " << avg_infer << " ms\n";
    if (is_embedding_mode) {
        std::cout << "Avg embedding: " << avg_service_embedding << " ms\n";
    } else if (looks_like_tracking) {
        std::cout << "Avg detect: " << avg_service_detect << " ms\n"
            << "Avg track: " << avg_service_track << " ms\n";
    } else {
        std::cout << "Avg preprocess: " << avg_service_preprocess << " ms\n"
            << "Avg model infer: " << avg_service_model_infer << " ms\n"
            << "Avg postprocess: " << avg_service_postprocess << " ms\n";
    }
    std::cout
        << "Draw executed/skipped: " << draw_executed_runs << "/" << draw_skipped_runs << "\n"
        << "Avg draw: " << avg_draw << " ms\n"
        << "FPS: " << fps << "\n";

    if (mode == BenchMode::kEmbeddingInfer && !args.image_path2.empty()) {
        cv::Mat image2 = cv::imread(args.image_path2);
        if (image2.empty()) {
            std::cerr << "Warning: could not read image2 for similarity: " << args.image_path2 << std::endl;
        } else {
            std::vector<float> embedding2;
            VisionServiceStatus st2 = service->InferEmbedding(image2, &embedding2);
            if (st2 == VISION_SERVICE_OK && !embedding.empty() && !embedding2.empty()) {
                const float sim = VisionService::EmbeddingSimilarity(embedding, embedding2);
                std::cout << "Image2: " << args.image_path2 << "\n"
                    << "Embedding similarity: " << sim << "\n";
            } else {
                std::cerr << "Warning: failed to infer image2 embedding: " << service->LastError() << std::endl;
            }
        }
    }

    return 0;
}
