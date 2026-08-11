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
                    const vision::Result* first) {
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
        const vision::BoundingBox box = vision::get_bbox(*first);
        out << "first_bbox=" << box.x1 << "," << box.y1 << ","
            << box.x2 << "," << box.y2 << "\n";
        out << "first_label=" << vision::get_label(*first) << "\n";
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

    VisionServiceResponse response;
    std::cout << "Infer: " << args.image_path
            << " conf_threshold=" << conf_threshold << std::endl;
    VisionServiceInferParams params;
    params.conf_threshold = conf_threshold;
    VisionServiceStatus status =
        service->Infer(args.image_path, &response, params);
    {
        const std::string msg = std::string("expected VISION_SERVICE_OK, got ") +
                                std::to_string(status) + ", LastError=" + service->LastError();
        check(status == VISION_SERVICE_OK, msg);
    }
    {
        const std::string msg = std::string("expected detection count > 0, got 0 (image=") +
                                args.image_path + ")";
        check(!response.results.empty(), msg);
    }

    cv::Mat image = cv::imread(args.image_path);
    check(!image.empty(), std::string("could not read image for bounds check: ") + args.image_path);
    const int cols = image.cols;
    const int rows = image.rows;

    float max_score = 0.0f;
    for (size_t i = 0; i < response.results.size(); ++i) {
        const auto& r = response.results[i];
        const vision::BoundingBox box = vision::get_bbox(r);
        const float score = vision::get_score(r);
        const int label = vision::get_label(r);
        if (score > max_score) {
            max_score = score;
        }
        {
            const std::string msg = std::string("label out of COCO range [0,79]: label=") +
                                    std::to_string(label) + " index=" + std::to_string(i);
            check(label >= 0 && label <= 79, msg);
        }
        {
            const std::string msg = std::string("score out of [conf,1]: score=") +
                                    std::to_string(score) + " conf=" +
                                    std::to_string(conf_threshold) + " index=" +
                                    std::to_string(i);
            check(score >= conf_threshold && score <= 1.0f, msg);
        }
        {
            const std::string msg = std::string("invalid bbox dimensions at index ") +
                                    std::to_string(i);
            check(box.x2 > box.x1 && box.y2 > box.y1, msg);
        }
        {
            std::string bbox = std::to_string(box.x1);
            bbox += ",";
            bbox += std::to_string(box.y1);
            bbox += ",";
            bbox += std::to_string(box.x2);
            bbox += ",";
            bbox += std::to_string(box.y2);
            const std::string img_size = std::to_string(cols) + "x" + std::to_string(rows);
            const std::string msg = std::string("bbox out of image bounds at index ") +
                                    std::to_string(i) + " bbox=(" + bbox + ") image=(" +
                                    img_size + ")";
            const bool in_bounds = box.x1 >= 0.0f && box.y1 >= 0.0f &&
                box.x2 <= static_cast<float>(cols) && box.y2 <= static_cast<float>(rows);
            check(in_bounds, msg);
        }
    }

    std::cout << "Invalid input branches on live service" << std::endl;
    VisionServiceResponse empty_response;
    status = service->Infer(cv::Mat(), &empty_response);
    {
        const std::string msg = std::string("empty Mat expected INVALID_ARGUMENT, got ") +
                                std::to_string(status);
        check(status == VISION_SERVICE_INVALID_ARGUMENT, msg);
    }

    cv::Mat gray(rows, cols, CV_8UC1, cv::Scalar(0));
    status = service->Infer(gray, &empty_response);
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

    const vision::Result* first = response.results.empty() ? nullptr : &response.results[0];
    write_artifact(args, response.results.size(), max_score, first);

    if (g_failures > 0) {
        std::cerr << g_failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: detections=" << response.results.size() << " max_score=" << max_score << std::endl;
    return 0;
}
