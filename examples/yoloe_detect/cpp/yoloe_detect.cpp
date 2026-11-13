/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <filesystem> // NOLINT(build/c++17)
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h> // NOLINT(build/include_order)

#include "vision_service.h"

namespace fs = std::filesystem;
namespace {
void usage(const char* program) {
    std::cout << "Usage: " << program << " CONFIG [options]\n"
                << "  --mode text|visual|fused (default: text)\n"
                << "  --image PATH           Target image (default: YAML test_image)\n"
                << "  --prompts person,bus   Text classes (default: YAML prompts)\n"
                << "  --visual-prompts PATH  Reference YAML (default: example "
                    "directory)\n"
                << "  --model-path PATH      Override detector ONNX\n"
                << "  --conf-threshold FLOAT Override confidence threshold\n"
                << "  --output PATH          Result (default: yoloe_detect_result.jpg)\n";
}
std::vector<std::string> split(const std::string& csv) {
    std::vector<std::string> names;
    std::istringstream stream(csv);
    std::string name;
    while (std::getline(stream, name, ',')) {
        const auto start = name.find_first_not_of(" \t");
        if (start == std::string::npos)
            throw std::invalid_argument("Empty class name");
        names.push_back(name.substr(start, name.find_last_not_of(" \t") - start + 1));
    }
    if (names.empty() || (!csv.empty() && csv.back() == ',')) {
        throw std::invalid_argument("Empty class name");
    }
    return names;
}
fs::path reference_path(std::string path, const fs::path& profile) {
    if (path.rfind("~/", 0) == 0) {
        const char* user_home = std::getenv("HOME");
        if (!user_home)
            throw std::runtime_error("HOME is unset");
        path = std::string(user_home) + path.substr(1);
    }
    const fs::path result(path);
    return result.is_absolute() ? result : profile.parent_path() / result;
}
std::vector<vision::VisualPrompt> load_visuals(const fs::path& path) {
    const auto data = YAML::LoadFile(path.string());
    if (!data.IsSequence() || data.size() == 0) {
        throw std::invalid_argument("Visual prompt YAML must be a nonempty list");
    }
    std::vector<vision::VisualPrompt> prompts;
    for (const auto& item : data) {
        vision::VisualPrompt prompt;
        prompt.class_name = item["class_name"].as<std::string>();
        const auto image = reference_path(item["image_path"].as<std::string>(), path);
        prompt.image = cv::imread(image.string());
        if (prompt.image.empty())
            throw std::runtime_error("Cannot read reference: " + image.string());
        const auto boxes = item["bboxes"];
        if (!boxes.IsSequence())
            throw std::invalid_argument("bboxes must be a list");
        for (const auto& box : boxes) {
            if (!box.IsSequence() || box.size() != 4) {
                throw std::invalid_argument("Each reference box needs 4 coordinates");
            }
            prompt.bboxes.push_back({box[0].as<float>(), box[1].as<float>(), box[2].as<float>(), box[3].as<float>()});
        }
        prompts.push_back(std::move(prompt));
    }
    return prompts;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    try {
        const fs::path config = fs::absolute(argv[1]);
        std::string mode = "text", image_path, model_path;
        fs::path visuals = config.parent_path().parent_path() / "visual_prompts.yaml";
        std::string output = "yoloe_detect_result.jpg";
        std::vector<std::string> texts;
        VisionServiceInferParams params;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                usage(argv[0]);
                return 0;
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + option);
            const std::string value = argv[++i];
            if (option == "--mode")
                mode = value;
            else if (option == "--image")
                image_path = value;
            else if (option == "--model-path")
                model_path = value;
            else if (option == "--output")
                output = value;
            else if (option == "--prompts")
                texts = split(value);
            else if (option == "--visual-prompts")
                visuals = fs::absolute(value);
            else if (option == "--conf-threshold")
                params.conf_threshold = std::stof(value);
            else
                throw std::invalid_argument("Unknown option: " + option);
        }
        if (mode != "text" && mode != "visual" && mode != "fused") {
            throw std::invalid_argument("--mode must be text, visual or fused");
        }
        if (mode == "visual" && !texts.empty()) {
            throw std::invalid_argument("Visual mode uses class_name from the reference YAML, not --prompts");
        }
        if (mode == "fused" && texts.empty()) {
            texts = YAML::LoadFile(config.string())["default_params"]["prompts"].as<std::vector<std::string>>();
            if (texts.empty())
                throw std::invalid_argument("Fused mode requires text classes");
        }
        auto service = VisionService::Create(config.string(), model_path, true);
        if (!service)
            throw std::runtime_error(VisionService::LastCreateError());
        if (image_path.empty())
            image_path = service->GetDefaultImage();
        VisionServiceRequest request;
        request.image = cv::imread(image_path);
        if (request.image.empty())
            throw std::runtime_error("Cannot read target image: " + image_path);
        request.params = params;
        request.prompts = texts;
        if (mode != "text")
            request.visual_prompts = load_visuals(visuals);
        VisionServiceResponse response;
        if (service->Infer(request, &response) != VISION_SERVICE_OK) {
            throw std::runtime_error(service->LastError());
        }
        const auto names = service->GetClassNames();
        std::cout << "Mode: " << mode << "; detected " << response.results.size() << " objects\n";
        for (const auto& result : response.results) {
            const auto& detection = std::get<vision::Detection>(result);
            std::cout << names.at(detection.label) << " " << std::fixed << std::setprecision(4) << detection.score
                        << " [" << detection.bbox.x1 << ", " << detection.bbox.y1 << ", " << detection.bbox.x2 << ", "
                        << detection.bbox.y2 << "]\n";
        }
        cv::Mat drawn;
        if (service->Draw(request.image, response, &drawn) != VISION_SERVICE_OK) {
            throw std::runtime_error(service->LastError());
        }
        if (!cv::imwrite(output, drawn))
            throw std::runtime_error("Cannot save: " + output);
        std::cout << "Result image saved to: " << output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
