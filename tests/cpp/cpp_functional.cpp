/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PR functional test: real Create + InferImage on yolov8n sample image.
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "vision_service.h"

namespace {

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    ++g_failures;
}

void check(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

float read_conf_threshold(const std::string& config_path, float fallback) {
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        if (config["default_params"] && config["default_params"]["conf_threshold"]) {
            return config["default_params"]["conf_threshold"].as<float>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not read conf_threshold from " << config_path
                << ": " << e.what() << ", using fallback " << fallback << std::endl;
    }
    return fallback;
}

struct Args {
    std::string config_path;
    std::string image_path;
    std::string output_path;
};

bool parse_args(int argc, char** argv, Args* args) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            args->config_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            args->image_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args->output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: vision_cpp_functional --config <yaml> --image <jpg> [--output <txt>]\n";
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    if (args->config_path.empty() || args->image_path.empty()) {
        std::cerr << "Error: --config and --image are required\n";
        return false;
    }
    return true;
}

void write_artifact(const Args& args,
                    size_t detection_count,
                    float max_score,
                    const VisionServiceResult* first) {
    if (args.output_path.empty()) {
        return;
    }
    std::ofstream out(args.output_path);
    if (!out) {
        fail("could not open artifact file: " + args.output_path);
        return;
    }
    out << "config=" << args.config_path << "\n";
    out << "image=" << args.image_path << "\n";
    out << "detection_count=" << detection_count << "\n";
    out << "max_score=" << max_score << "\n";
    if (first != nullptr) {
        out << "first_bbox=" << first->x1 << "," << first->y1 << ","
            << first->x2 << "," << first->y2 << "\n";
        out << "first_label=" << first->label << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) {
        return 1;
    }

    const float conf_threshold = read_conf_threshold(args.config_path, 0.25f);

    std::cout << "Create config: " << args.config_path << std::endl;
    auto service = VisionService::Create(args.config_path);
    if (!service) {
        const std::string msg = std::string("Create returned nullptr, LastCreateError=") +
                                VisionService::LastCreateError();
        fail(msg);
        return g_failures > 0 ? 1 : 0;
    }
    {
        const std::string err = VisionService::LastCreateError();
        const std::string msg = std::string("expected empty LastCreateError after Create, got: ") +
                                err;
        check(err.empty(), msg);
    }

    std::vector<VisionServiceResult> results;
    std::cout << "InferImage: " << args.image_path
            << " conf_threshold=" << conf_threshold << std::endl;
    VisionServiceStatus status =
        service->InferImage(args.image_path, &results, conf_threshold);
    {
        const std::string msg = std::string("expected VISION_SERVICE_OK, got ") +
                                std::to_string(status) + ", LastError=" + service->LastError();
        check(status == VISION_SERVICE_OK, msg);
    }
    {
        const std::string msg = std::string("expected detection count > 0, got 0 (image=") +
                                args.image_path + ")";
        check(!results.empty(), msg);
    }

    cv::Mat image = cv::imread(args.image_path);
    check(!image.empty(), std::string("could not read image for bounds check: ") + args.image_path);
    const int cols = image.cols;
    const int rows = image.rows;

    float max_score = 0.0f;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (r.score > max_score) {
            max_score = r.score;
        }
        {
            const std::string msg = std::string("label out of COCO range [0,79]: label=") +
                                    std::to_string(r.label) + " index=" + std::to_string(i);
            check(r.label >= 0 && r.label <= 79, msg);
        }
        {
            const std::string msg = std::string("score out of [conf,1]: score=") +
                                    std::to_string(r.score) + " conf=" +
                                    std::to_string(conf_threshold) + " index=" +
                                    std::to_string(i);
            check(r.score >= conf_threshold && r.score <= 1.0f, msg);
        }
        {
            const std::string msg = std::string("invalid bbox dimensions at index ") +
                                    std::to_string(i);
            check(r.x2 > r.x1 && r.y2 > r.y1, msg);
        }
        {
            std::string bbox = std::to_string(r.x1);
            bbox += ",";
            bbox += std::to_string(r.y1);
            bbox += ",";
            bbox += std::to_string(r.x2);
            bbox += ",";
            bbox += std::to_string(r.y2);
            const std::string img_size = std::to_string(cols) + "x" + std::to_string(rows);
            const std::string msg = std::string("bbox out of image bounds at index ") +
                                    std::to_string(i) + " bbox=(" + bbox + ") image=(" +
                                    img_size + ")";
            const bool in_bounds = r.x1 >= 0.0f && r.y1 >= 0.0f &&
                r.x2 <= static_cast<float>(cols) && r.y2 <= static_cast<float>(rows);
            check(in_bounds, msg);
        }
    }

    std::cout << "Invalid input branches on live service" << std::endl;
    std::vector<VisionServiceResult> empty_results;
    status = service->InferImage(cv::Mat(), &empty_results);
    {
        const std::string msg = std::string("empty Mat expected INVALID_ARGUMENT, got ") +
                                std::to_string(status);
        check(status == VISION_SERVICE_INVALID_ARGUMENT, msg);
    }

    cv::Mat gray(rows, cols, CV_8UC1, cv::Scalar(0));
    status = service->InferImage(gray, &empty_results);
    {
        const std::string msg = std::string("1-channel Mat expected INVALID_ARGUMENT, got ") +
                                std::to_string(status);
        check(status == VISION_SERVICE_INVALID_ARGUMENT, msg);
    }

    std::cout << "EmbeddingSimilarity pure function" << std::endl;
    const std::vector<float> a = {1.0f, 0.0f, 0.0f};
    const std::vector<float> b = {-1.0f, 0.0f, 0.0f};
    const std::vector<float> c = {0.0f, 1.0f, 0.0f};
    const float sim_aa = VisionService::EmbeddingSimilarity(a, a);
    const float sim_ab = VisionService::EmbeddingSimilarity(a, b);
    const float sim_ac = VisionService::EmbeddingSimilarity(a, c);
    {
        const std::string msg = std::string("EmbeddingSimilarity(a,a) expected ~1.0, got ") +
                                std::to_string(sim_aa);
        check(std::abs(sim_aa - 1.0f) < 1e-5f, msg);
    }
    {
        const std::string msg = std::string("EmbeddingSimilarity(a,b) expected ~-1.0, got ") +
                                std::to_string(sim_ab);
        check(std::abs(sim_ab - (-1.0f)) < 1e-5f, msg);
    }
    {
        const std::string msg = std::string("EmbeddingSimilarity(a,c) expected ~0.0, got ") +
                                std::to_string(sim_ac);
        check(std::abs(sim_ac - 0.0f) < 1e-5f, msg);
    }

    const VisionServiceResult* first = results.empty() ? nullptr : &results[0];
    write_artifact(args, results.size(), max_score, first);

    if (g_failures > 0) {
        std::cerr << g_failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: detections=" << results.size() << " max_score=" << max_score << std::endl;
    return 0;
}
