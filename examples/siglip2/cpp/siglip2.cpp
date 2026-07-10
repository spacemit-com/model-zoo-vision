/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <fstream>
#include <iostream>  // NOLINT(build/include_order)
#include <iomanip>   // NOLINT(build/include_order)
#include <memory>    // NOLINT(build/include_order)
#include <sstream>   // NOLINT(build/include_order)
#include <string>    // NOLINT(build/include_order)
#include <variant>   // NOLINT(build/include_order)
#include <vector>    // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

static std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto start = item.find_first_not_of(" \t");
        const auto end = item.find_last_not_of(" \t");
        if (start == std::string::npos) {
            continue;
        }
        out.push_back(item.substr(start, end - start + 1));
    }
    return out;
}

static std::vector<std::string> load_labels_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + path);
    }
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        labels.push_back(line);
    }
    return labels;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
            << "  Options: --image <path> --labels <scene_labels.txt>\n"
            << "           --text \"label1,label2\" --topk N --model-path <path>"
            << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path;
    std::string labels_path;
    std::string text_csv;
    int topk = 5;
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            labels_path = argv[++i];
        } else if (arg == "--text" && i + 1 < argc) {
            text_csv = argv[++i];
        } else if (arg == "--topk" && i + 1 < argc) {
            topk = std::stoi(argv[++i]);
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
        }
    }

    std::unique_ptr<VisionService> service =
        VisionService::Create(config_path, model_path_override, true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    if (image_path.empty()) {
        image_path = service->GetDefaultImage();
    }
    if (image_path.empty()) {
        std::cerr << "Error: No image path. Use --image or set test_image in config." << std::endl;
        return 1;
    }

    std::vector<std::string> labels;
    if (!text_csv.empty()) {
        labels = split_csv(text_csv);
    } else {
        if (labels_path.empty()) {
            labels_path = service->GetConfigPathValue("scene_labels_path");
        }
        if (!labels_path.empty()) {
            try {
                labels = load_labels_file(labels_path);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                return 1;
            }
        }
    }
    if (labels.empty()) {
        labels = {"a photo of a dog", "a photo of a cat", "a photo of a car"};
    }

    VisionServiceResponse response;
    if (service->Infer(image_path, &response) != VISION_SERVICE_OK) {
        std::cerr << "Error: " << service->LastError() << std::endl;
        return 1;
    }
    if (response.results.empty()) {
        std::cerr << "Error: No image embedding returned." << std::endl;
        return 1;
    }
    const vision::Embedding* image_emb = std::get_if<vision::Embedding>(&response.results[0]);
    if (image_emb == nullptr) {
        std::cerr << "Error: Model did not return an embedding." << std::endl;
        return 1;
    }

    std::vector<std::pair<std::string, float>> scored;
    scored.reserve(labels.size());
    std::cout << "Image: " << image_path << std::endl;
    for (const std::string& label : labels) {
        std::vector<float> text_emb;
        if (service->EncodeText(label, &text_emb) != VISION_SERVICE_OK) {
            std::cerr << "Error encoding text: " << service->LastError() << std::endl;
            return 1;
        }
        const float score = VisionService::EmbeddingSimilarity(image_emb->embedding, text_emb);
        scored.emplace_back(label, score);
    }

    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    const int show_k = std::min(topk, static_cast<int>(scored.size()));
    std::cout << "Top " << show_k << " matches:" << std::endl;
    for (int i = 0; i < show_k; ++i) {
        std::cout << "  " << scored[static_cast<size_t>(i)].first << " : "
            << std::fixed << std::setprecision(4) << scored[static_cast<size_t>(i)].second
            << std::endl;
    }
    return 0;
}
