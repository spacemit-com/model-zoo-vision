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
    std::string model_path_override;
    int runs = 100;
    int warmup = 5;
    bool verbose_timing = false;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
        << " --config <yaml> [--image <path>] [--model-path <path>] [--runs N] "
        << "[--warmup N] [--verbose-timing]\n"
        << "Example: " << prog
        << " --config examples/yolov8/config/yolov8.yaml\n"
        << "Note: if --image is omitted, benchmark uses test_image from config.\n";
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
        args.image_path = service->GetDefaultImage();
        if (args.image_path.empty()) {
            std::cerr << "Error: --image is not set and config has no valid test_image" << std::endl;
            return 1;
        }
    }

    cv::Mat image = cv::imread(args.image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read image: " << args.image_path << std::endl;
        return 1;
    }

    std::vector<VisionServiceResult> results;

    for (int i = 0; i < args.warmup; ++i) {
        VisionServiceStatus st = service->InferImage(image, &results);
        if (st != VISION_SERVICE_OK) {
            std::cerr << "Error: warmup InferImage failed: " << service->LastError() << std::endl;
            return 1;
        }
    }

    double infer_total = 0.0;
    double draw_total = 0.0;
    double service_infer_total = 0.0;
    double service_draw_total = 0.0;
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
        const auto t0 = std::chrono::steady_clock::now();
        VisionServiceStatus st = service->InferImage(image, &results);
        const auto t1 = std::chrono::steady_clock::now();
        if (st != VISION_SERVICE_OK) {
            std::cerr << "Error: InferImage failed: " << service->LastError() << std::endl;
            return 1;
        }

        infer_total += to_ms(t1 - t0);
        VisionServiceTiming timing_after_infer = service->GetLastTiming();
        service_infer_total += timing_after_infer.infer_ms;
        service_preprocess_total += timing_after_infer.preprocess_ms;
        service_model_infer_total += timing_after_infer.model_infer_ms;
        service_postprocess_total += timing_after_infer.postprocess_ms;
        service_detect_total += timing_after_infer.detect_ms;
        service_track_total += timing_after_infer.track_ms;

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
            } else {
                ++draw_executed_runs;
                draw_total += to_ms(d1 - d0);
                service_draw_total += service->GetLastTiming().draw_ms;
            }
        }
    }

    const double avg_infer = infer_total / args.runs;
    const double avg_service_infer = service_infer_total / args.runs;
    const double avg_service_preprocess = service_preprocess_total / args.runs;
    const double avg_service_model_infer = service_model_infer_total / args.runs;
    const double avg_service_postprocess = service_postprocess_total / args.runs;
    const double avg_service_detect = service_detect_total / args.runs;
    const double avg_service_track = service_track_total / args.runs;
    const double avg_draw = (draw_executed_runs > 0) ? (draw_total / draw_executed_runs) : 0.0;
    const double avg_service_draw =
        (draw_executed_runs > 0) ? (service_draw_total / draw_executed_runs) : 0.0;
    const double fps = (avg_infer > 0.0) ? (1000.0 / avg_infer) : 0.0;
    const bool looks_like_tracking = (avg_service_detect > 0.0) || (avg_service_track > 0.0);
    std::string stage_timing;
    if (looks_like_tracking) {
        stage_timing =
            "Avg detect (service timing): " + std::to_string(avg_service_detect) + " ms\n" +
            "Avg track (service timing): " + std::to_string(avg_service_track) + " ms\n";
    } else {
        stage_timing =
            "Avg preprocess (service timing): " + std::to_string(avg_service_preprocess) + " ms\n" +
            "Avg model infer (service timing): " + std::to_string(avg_service_model_infer) + " ms\n" +
            "Avg postprocess (service timing): " + std::to_string(avg_service_postprocess) + " ms\n";
    }

    std::cout << "Config: " << args.config_path << "\n"
        << "Image: " << args.image_path << "\n"
        << "Runs: " << args.runs << " (warmup " << args.warmup << ")\n"
        << "Avg infer (external benchmark): " << avg_infer << " ms\n"
        << "Avg infer (service timing): " << avg_service_infer << " ms\n"
        << stage_timing
        << "Draw executed/skipped: " << draw_executed_runs << "/" << draw_skipped_runs << "\n"
        << "Avg draw (external benchmark, executed only): " << avg_draw << " ms\n"
        << "Avg draw (service timing, executed only): " << avg_service_draw << " ms\n"
        << "FPS (avg external infer): " << fps << "\n";

    return 0;
}
