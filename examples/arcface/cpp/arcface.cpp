/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>    // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <variant>     // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

static const float kSimilarityThreshold = 0.6f;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
                    << "  Options (any order): --model-path <path> --image1 <path> --image2 <path> --threshold <f>\n"
                    << "  Default images from config: test_image1, test_image2" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image1_path;
    std::string image2_path;
    float threshold = kSimilarityThreshold;
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image1" && i + 1 < argc) {
            image1_path = argv[++i];
        } else if (arg == "--image2" && i + 1 < argc) {
            image2_path = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        }
    }

    std::unique_ptr<VisionService> service = VisionService::Create(
        config_path,
        model_path_override,
        true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    if (image1_path.empty()) {
        const std::string p = service->GetConfigPathValue("test_image1");
        if (!p.empty()) image1_path = p;
    }
    if (image2_path.empty()) {
        const std::string p = service->GetConfigPathValue("test_image2");
        if (!p.empty()) image2_path = p;
    }

    if (image1_path.empty() || image2_path.empty()) {
        std::cerr << "Error: Need two face images. Use --image1/--image2 or set test_image1/test_image2 in config."
                    << std::endl;
        return 1;
    }

    std::vector<float> emb1;
    std::vector<float> emb2;

    auto extract_embedding = [&](const std::string& path, std::vector<float>* out) -> int {
        VisionServiceResponse response;
        int ret = service->Infer(path, &response);
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
    };

    int r1 = extract_embedding(image1_path, &emb1);
    int r2 = extract_embedding(image2_path, &emb2);

    if (r1 != VISION_SERVICE_OK || r2 != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << std::endl;
        return 1;
    }

    if (emb1.size() != emb2.size()) {
        std::cerr << "Error: embedding dimension mismatch " << emb1.size() << " vs " << emb2.size() << std::endl;
        return 1;
    }

    float similarity = VisionService::EmbeddingSimilarity(emb1, emb2);
    std::cout << "相似度: " << std::fixed << std::setprecision(4) << similarity << std::endl;
    if (similarity >= threshold) {
        std::cout << "判断: 同一人" << std::endl;
    } else {
        std::cout << "判断: 非同一人" << std::endl;
    }

    return 0;
}
