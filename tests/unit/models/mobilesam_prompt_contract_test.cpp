/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_service.h"  // NOLINT(build/include_subdir)

namespace {

struct TestConfig {
    std::filesystem::path directory;
    std::filesystem::path path;

    TestConfig() {
        auto pattern = (std::filesystem::temp_directory_path() /
            "mobilesam-contract-XXXXXX").string();
        if (!mkdtemp(pattern.data())) {
            throw std::runtime_error("Could not create test directory");
        }
        directory = pattern;
        path = directory / "config.yaml";
        std::ofstream config(path);
        config << "class: deploy.mobilesam1.MobileSAMSegmentor\n"
            << "model_path: missing-encoder.onnx\n"
            << "default_params:\n"
            << "  decoder_model_path: missing-decoder.onnx\n"
            << "  providers: [CPUExecutionProvider]\n";
        config.close();
        if (!config) throw std::runtime_error("Could not write test config");
    }

    ~TestConfig() {
        std::error_code error;
        std::filesystem::remove(path, error);
        std::filesystem::remove(directory, error);
    }
};

}  // namespace

int main() {
    try {
        TestConfig config;
        auto service = VisionService::Create(config.path.string(), "", true);
        if (!service) {
            throw std::runtime_error(VisionService::LastCreateError());
        }
        int failures = 0;
        const auto reject = [&](const std::vector<cv::Point2f>& points,
                                const std::vector<int>& labels,
                                const std::string& expected) {
            VisionServiceRequest request{};
            request.image = cv::Mat::zeros(32, 32, CV_8UC3);
            request.point_coords = points;
            request.point_labels = labels;
            VisionServiceResponse response;
            // A reused response must not expose stale successful results.
            response.results.emplace_back(vision::Segmentation{});
            const auto status = service->Infer(request, &response);
            if (status != VISION_SERVICE_INFER_FAILED ||
                !response.results.empty() ||
                response.error_message.find(expected) == std::string::npos ||
                service->LastError() != response.error_message) {
                std::cerr << "FAIL: expected " << expected << ", got "
                    << response.error_message << '\n';
                ++failures;
            }
        };
        reject({}, {}, "requires matching points and labels");
        reject({{5, 5}}, {}, "requires matching points and labels");
        reject({{5, 5}}, {1, 0}, "requires matching points and labels");
        reject({}, {1}, "requires matching points and labels");
        reject({{5, 5}}, {4}, "invalid point or label");
        reject({{5, 5}}, {-2}, "invalid point or label");
        reject({{std::numeric_limits<float>::quiet_NaN(), 5}}, {1},
            "invalid point or label");
        reject({{5, std::numeric_limits<float>::infinity()}}, {1},
            "invalid point or label");
        reject({{5, 5}}, {2}, "invalid box corners");
        reject({{5, 5}}, {3}, "missing top-left box corner");
        reject({{5, 5}, {10, 10}}, {3, 2}, "missing top-left box corner");
        reject({{5, 5}, {10, 10}}, {2, 1}, "invalid box corners");
        reject({{5, 5}, {5, 10}}, {2, 3}, "invalid box corners");
        reject({{5, 5}, {10, 5}}, {2, 3}, "invalid box corners");
        reject({{10, 10}, {5, 5}}, {2, 3}, "invalid box corners");
        if (failures) return 1;
        std::cout << "PASS: MobileSAM public prompt contract (15 cases)\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
