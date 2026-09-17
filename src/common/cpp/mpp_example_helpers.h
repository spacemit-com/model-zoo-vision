/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MPP_EXAMPLE_HELPERS_H
#define MPP_EXAMPLE_HELPERS_H

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include "mpp_frame_source.h"
#include "vision_service.h"

namespace vision_mpp {

// Demo-only options. Model configuration and VisionService's public API do
// not depend on this parser. Zero requested dimensions preserve OpenCV defaults.
struct ExampleInputConfig {
    bool use_camera = false;
    std::string image_path;
    MppFrameSourceConfig camera;
    int width = 0;
    int height = 0;
    int fps = 0;
};

inline int ParseInputInteger(const std::string& value, const std::string& key,
                            int minimum = 1) {
    size_t consumed = 0;
    int number;
    try {
        number = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(key + " must be an integer");
    }
    if (consumed != value.size() || number < minimum) {
        throw std::invalid_argument(key + " has an invalid value: " + value);
    }
    return number;
}

inline ExampleInputConfig ParseExampleInput(
    const std::string& config_path, int argc, char** argv) {
    ExampleInputConfig result;
    const YAML::Node config = YAML::LoadFile(config_path);
    const YAML::Node input = config["input"];
    std::string backend = "opencv";
    std::string source;
    if (input) {
        if (!input.IsMap()) throw std::invalid_argument("input must be a map");
        if (!input["type"] || !input["type"].IsScalar()) {
            throw std::invalid_argument("input.type must be image or camera");
        }
        const std::string type = input["type"].as<std::string>();
        if (type != "image" && type != "camera") {
            throw std::invalid_argument("input.type must be image or camera");
        }
        result.use_camera = type == "camera";
        if (input["source"]) source = input["source"].as<std::string>();
        if (input["backend"]) backend = input["backend"].as<std::string>();
        if (backend != "opencv" && backend != "mpp") {
            throw std::invalid_argument("input.backend must be opencv or mpp");
        }
        for (const char* key : {"width", "height", "fps"}) {
            if (!input[key]) continue;
            if (!result.use_camera) {
                throw std::invalid_argument(std::string("input.") + key +
                                            " is only valid for camera input");
            }
            const int value = ParseInputInteger(input[key].as<std::string>(), key);
            if (std::string(key) == "width") result.width = value;
            if (std::string(key) == "height") result.height = value;
            if (std::string(key) == "fps") result.fps = value;
        }
        if (result.use_camera) {
            result.camera.v4l2_dev = source;
        } else if (!source.empty()) {
            if (source.compare(0, 2, "~/") == 0) {
                const char* user_home = std::getenv("HOME");
                if (!user_home) throw std::invalid_argument("HOME is not set");
                source = std::string(user_home) + source.substr(1);
            }
            std::filesystem::path path(source);
            if (path.is_relative()) {
                path = std::filesystem::absolute(config_path).parent_path() / path;
            }
            result.image_path = path.lexically_normal().string();
        }
    }

    bool image_arg = false;
    bool camera_arg = false;
    bool mpp_arg = false;
    bool camera_option = false;
    bool id_arg = false;
    bool device_arg = false;
    std::string device;
    // Skip values of unrelated existing demo options too: an option-looking
    // filename must not accidentally select a camera backend.
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc || std::string(argv[i]).empty()) {
                throw std::invalid_argument("missing value for " + arg);
            }
            return argv[i];
        };
        if (arg == "--image") {
            result.image_path = value();
            image_arg = true;
        } else if (arg == "--use-camera") {
            camera_arg = true;
        } else if (arg == "--camera-id") {
            result.camera.camera_id = ParseInputInteger(value(), arg, 0);
            id_arg = camera_option = true;
        } else if (arg == "--v4l2-dev") {
            device = value();
            device_arg = camera_option = true;
        } else if (arg == "--use-mpp") {
            mpp_arg = camera_option = true;
        } else if (arg == "--mpp-width") {
            result.width = ParseInputInteger(value(), arg);
            camera_option = true;
        } else if (arg == "--mpp-height") {
            result.height = ParseInputInteger(value(), arg);
            camera_option = true;
        } else if (arg == "--mpp-fps") {
            result.fps = ParseInputInteger(value(), arg);
            camera_option = true;
        } else if (arg == "--mpp-timeout") {
            result.camera.timeout_ms = ParseInputInteger(value(), arg);
            camera_option = true;
        } else if (arg == "--mpp-format") {
            result.camera.format = value();
            camera_option = true;
        } else if (arg == "--mpp-chn") {
            result.camera.vi_chn = ParseInputInteger(value(), arg, 0);
            camera_option = true;
        } else if (arg == "--mpp-vi") {
            result.camera.use_vi = true;
            camera_option = true;
        } else if (arg == "--mpp-cpu-color") {
            result.camera.cpu_color = true;
            camera_option = true;
        } else if (arg == "--model-path" || arg == "--output") {
            (void)value();
        } else if (arg != "--help") {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (image_arg && (camera_arg || camera_option)) {
        throw std::invalid_argument("--image conflicts with camera options");
    }
    if (image_arg) {
        result.use_camera = false;
        backend = "opencv";
    } else if (camera_arg) {
        if (!result.use_camera) backend = "opencv";
        result.use_camera = true;
    }
    if (!result.use_camera && camera_option) {
        throw std::invalid_argument("camera options require camera input");
    }
    if (id_arg) result.camera.v4l2_dev.clear();
    if (device_arg) result.camera.v4l2_dev = device;
    if (mpp_arg) backend = "mpp";
    if (!result.use_camera && backend != "opencv") {
        throw std::invalid_argument("image input only supports opencv");
    }
    result.camera.use_mpp = backend == "mpp";
    if (result.width) result.camera.width = result.width;
    if (result.height) result.camera.height = result.height;
    if (result.fps) result.camera.fps = result.fps;
    if (result.use_camera && !result.camera.v4l2_dev.empty() &&
        result.camera.v4l2_dev.compare(0, 5, "/dev/") != 0) {
        throw std::invalid_argument("camera source must be a /dev/ device path");
    }
    if (result.camera.format != "MJPEG" && result.camera.format != "YUYV" &&
        result.camera.format != "NV12") {
        throw std::invalid_argument("MPP format must be MJPEG, YUYV, or NV12");
    }
    return result;
}

// Existing demos keep their own business/video/positional argument parser.
// Only pass already parsed input overrides here, so unrelated options survive.
inline bool ConfigureLegacyDemoInput(
    const std::string& config_path, bool camera_requested, int camera_id,
    bool camera_id_set, const std::string& media_override, bool video_demo,
    bool supports_mpp, ExampleInputConfig* output) {
    try {
        std::vector<std::string> args = {"demo", config_path};
        if (camera_requested) {
            args.push_back("--use-camera");
        } else if (!media_override.empty()) {
            // A legacy video path also overrides configured camera selection.
            args.insert(args.end(), {"--image", media_override});
        }
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        *output = ParseExampleInput(config_path, argv.size(), argv.data());
        // Legacy demos allow camera-id alongside video/image arguments; it
        // only affects capture when camera mode is selected.
        if (camera_id_set) {
            output->camera.camera_id = camera_id;
            output->camera.v4l2_dev.clear();
        }
        if (video_demo && !camera_requested && media_override.empty()) {
            const auto input = YAML::LoadFile(config_path)["input"];
            if (input && input["type"].as<std::string>() == "image") {
                std::cerr << "Image input is not supported by this video/camera demo.\n";
                return false;
            }
        }
        if (output->camera.use_mpp && !supports_mpp) {
            std::cerr << "MPP camera backend is not supported by this demo. "
                << "Use backend: opencv instead.\n";
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        std::cerr << "Invalid input configuration: " << error.what() << '\n';
        return false;
    }
}

inline bool OpenExampleCamera(cv::VideoCapture* cap, const ExampleInputConfig& input) {
    const auto& cfg = input.camera;
    if (cfg.v4l2_dev.empty()) cap->open(cfg.camera_id);
    else cap->open(cfg.v4l2_dev);
    if (!cap->isOpened()) return false;
    const auto set = [&](int property, int value, const char* name) {
        if (value && !cap->set(property, value)) {
            std::cerr << "Warning: camera rejected requested " << name << '=' << value << '\n';
        }
    };
    set(cv::CAP_PROP_FRAME_WIDTH, input.width, "width");
    set(cv::CAP_PROP_FRAME_HEIGHT, input.height, "height");
    set(cv::CAP_PROP_FPS, input.fps, "fps");
    std::cout << "OpenCV camera reports " << cap->get(cv::CAP_PROP_FRAME_WIDTH)
        << 'x' << cap->get(cv::CAP_PROP_FRAME_HEIGHT)
        << " at " << cap->get(cv::CAP_PROP_FPS) << " FPS\n";
    return true;
}

inline bool BuildVisionRequest(
    const MppFrame& frame,
    VisionServiceRequest* request)
{
    if (request == nullptr || frame.empty()) return false;
    request->image = frame.image();
    request->image_format =
        frame.pixel_format() == MppFramePixelFormat::kNv12
        ? VisionPixelFormat::NV12
        : VisionPixelFormat::BGR8;
    request->image_dma_fd = frame.dma_fd();
    return true;
}

inline std::string FindImagePreprocessBackend(
    const VisionServiceProfile& profile)
{
    for (const auto& component : profile.components) {
        if (component.name == "image_preprocess.v2d") {
            return "v2d";
        }
        if (component.name == "image_preprocess.opencl") {
            return "opencl";
        }
        if (component.name == "image_preprocess.cpu") {
            return "cpu";
        }
    }
    return {};
}

inline bool ParseMppArgs(int argc, char** argv, int camera_id, MppFrameSourceConfig* cfg) {
    if (cfg == nullptr) return false;
    cfg->camera_id = camera_id;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--use-mpp") {
            cfg->use_mpp = true;
        } else if (arg == "--mpp-vi") {
            cfg->use_vi = true;
        } else if (arg == "--mpp-format" && i + 1 < argc) {
            cfg->format = argv[++i];
        } else if (arg == "--mpp-width" && i + 1 < argc) {
            cfg->width = std::atoi(argv[++i]);
        } else if (arg == "--mpp-height" && i + 1 < argc) {
            cfg->height = std::atoi(argv[++i]);
        } else if (arg == "--mpp-fps" && i + 1 < argc) {
            cfg->fps = std::atoi(argv[++i]);
        } else if (arg == "--mpp-timeout" && i + 1 < argc) {
            cfg->timeout_ms = std::atoi(argv[++i]);
        } else if (arg == "--v4l2-dev" && i + 1 < argc) {
            cfg->v4l2_dev = argv[++i];
        } else if (arg == "--mpp-chn" && i + 1 < argc) {
            cfg->vi_chn = std::atoi(argv[++i]);
        } else if (arg == "--mpp-cpu-color") {
            cfg->cpu_color = true;
        }
    }
    return cfg->use_mpp;
}

inline const char* MppUsage() {
    return "  --use-mpp             Use MPP camera backend instead of cv::VideoCapture\n"
            "  --mpp-vi              Use MPP VI/ISP path (default UVC)\n"
            "  --mpp-format <fmt>    UVC pixel format: MJPEG (default) | YUYV | NV12\n"
            "  --mpp-width <w>       MPP capture width (default 640)\n"
            "  --mpp-height <h>      MPP capture height (default 480)\n"
            "  --mpp-fps <f>         MPP capture fps (default 30)\n"
            "  --mpp-timeout <ms>    MPP frame timeout ms (default 1000)\n"
            "  --v4l2-dev <path>     Override /dev/video<camera_id>\n"
            "  --mpp-chn <id>        VI channel id (with --mpp-vi, default 0)\n"
            "  --mpp-cpu-color       Force CPU cvtColor for NV12->BGR (skip V2D)\n";
}

}  // namespace vision_mpp

#endif  // MPP_EXAMPLE_HELPERS_H
