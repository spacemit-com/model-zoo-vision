/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <fstream>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

#include "mpp_example_helpers.h"

namespace {

void check_input_options() {
    struct ConfigFile {
        char path[64] = "/tmp/vision-example-input-XXXXXX";
        ConfigFile() {
            const int fd = ::mkstemp(path);
            if (fd < 0) throw std::runtime_error("cannot create test config");
            ::close(fd);
        }
        ~ConfigFile() { std::remove(path); }
    } file;
    auto parse = [&](const std::string& yaml, std::vector<std::string> args = {}) {
        std::ofstream config(file.path);
        config << yaml;
        config.close();
        std::vector<std::string> values = {"demo", file.path};
        values.insert(values.end(), args.begin(), args.end());
        std::vector<char*> argv;
        for (auto& value : values) argv.push_back(value.data());
        return vision_mpp::ParseExampleInput(file.path, argv.size(), argv.data());
    };
    auto require = [](bool ok) {
        if (!ok) throw std::runtime_error("input configuration assertion failed");
    };
    auto rejects = [&](const std::string& yaml, std::vector<std::string> args = {}) {
        bool rejected = false;
        try { (void)parse(yaml, args); }
        catch (const std::exception&) { rejected = true; }
        require(rejected);
    };
    auto input = parse("test_image: old.jpg\n");
    require(!input.use_camera && input.image_path.empty() && input.width == 0);
    input = parse("{}", {"--use-camera"});
    require(input.use_camera && !input.camera.use_mpp);
    input = parse("{}", {"--use-camera", "--use-mpp"});
    require(input.use_camera && input.camera.use_mpp);
    rejects("{}", {"--use-mpp"});
    rejects("{}", {"--no-display"});
    rejects("{}", {"--video", "a.mp4"});

    const std::string camera_defaults =
        "camera:\n  backend: mpp\n  device: /dev/video2\n"
        "  width: 1280\n  height: 720\n  fps: 30\n";
    input = parse(camera_defaults);
    require(!input.use_camera && !input.camera.use_mpp && input.image_path.empty());
    input = parse(camera_defaults, {"--image", "photo.jpg"});
    require(!input.use_camera && !input.camera.use_mpp && input.image_path == "photo.jpg");
    input = parse(camera_defaults, {"--use-camera"});
    require(input.use_camera && input.camera.use_mpp &&
        input.camera.v4l2_dev == "/dev/video2" && input.camera.width == 1280 &&
        input.camera.height == 720 && input.camera.fps == 30);
    input = parse(camera_defaults,
        {"--use-camera", "--camera-id", "5", "--mpp-width", "640", "--mpp-fps", "25"});
    require(input.camera.v4l2_dev.empty() && input.camera.camera_id == 5 &&
        input.width == 640 && input.height == 720 && input.fps == 25);
    input = parse("camera: {backend: opencv, width: 800}\n", {"--use-camera"});
    require(input.use_camera && !input.camera.use_mpp && input.width == 800);
    input = parse(camera_defaults, {"--use-camera", "--v4l2-dev", "/dev/video4"});
    require(input.camera.v4l2_dev == "/dev/video4");
    rejects("camera: invalid\n");
    rejects("camera: {backend: invalid}\n");
    rejects("camera: {width: 0}\n");
    rejects("camera: {device: 'rtsp://example'}\n", {"--use-camera"});
    rejects(camera_defaults, {"--image", "a.jpg", "--use-camera"});
    rejects(camera_defaults, {"--use-camera", "--image", "a.jpg"});
    rejects(camera_defaults, {"--use-camera", "--camera-id", "-1"});
    rejects(camera_defaults, {"--use-camera", "--mpp-width", "640oops"});
    rejects(camera_defaults, {"--use-camera", "--mpp-fps"});
    // Obsolete input.type no longer selects a camera or supplies an image.
    input = parse("input: {type: camera, source: /dev/video9}\n");
    require(!input.use_camera && !input.camera.use_mpp && input.camera.v4l2_dev.empty());

    // Existing application/tracking parsers retain their business arguments
    // and pass only resolved input overrides to the common configuration code.
    auto legacy = [&](const std::string& yaml, bool camera_requested,
                        bool id_set, const std::string& media,
                        bool mpp_supported) {
        std::ofstream config(file.path);
        config << yaml;
        config.close();
        return vision_mpp::ConfigureLegacyDemoInput(
            file.path, camera_requested, 3, id_set, media,
            mpp_supported, &input);
    };
    require(legacy("{}", false, false, "", true));
    require(!input.use_camera && input.image_path.empty());
    require(legacy(camera_defaults, false, false, "", false));
    require(!input.use_camera && !input.camera.use_mpp);
    require(legacy(camera_defaults, false, false, "", true));
    require(!input.use_camera && !input.camera.use_mpp);
    require(!legacy(camera_defaults, true, false, "", false));
    require(legacy(camera_defaults, true, true, "", true));
    require(input.use_camera && input.camera.use_mpp &&
        input.camera.v4l2_dev.empty() && input.camera.camera_id == 3);
    require(legacy(camera_defaults, false, true, "clip.mp4", true));
    require(!input.use_camera && !input.camera.use_mpp && input.image_path == "clip.mp4");
}

}  // namespace

int main()
{
    try {
        check_input_options();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    using vision_mpp::MppFrame;
    using vision_mpp::MppFrameSource;

    static_assert(std::is_default_constructible_v<MppFrame>);
    static_assert(std::is_move_constructible_v<MppFrame>);
    static_assert(std::is_move_assignable_v<MppFrame>);
    static_assert(!std::is_copy_constructible_v<MppFrame>);
    static_assert(!std::is_copy_assignable_v<MppFrame>);

    using ReadFrame = bool (MppFrameSource::*)(MppFrame*);
    using ConvertFrame = bool (MppFrameSource::*)(
        const MppFrame&, cv::Mat*);
    const ReadFrame read_frame =
        static_cast<ReadFrame>(&MppFrameSource::read);
    const ConvertFrame convert_frame = &MppFrameSource::to_bgr;
    (void)read_frame;
    (void)convert_frame;

    MppFrame empty;
    if (!empty.empty() || empty.dma_fd() != -1) {
        std::cerr << "FAIL: default MPP frame is not empty\n";
        return 1;
    }

    MppFrame moved(std::move(empty));
    if (!moved.empty()) {
        std::cerr << "FAIL: moved default MPP frame is not empty\n";
        return 1;
    }
    VisionServiceRequest request;
    if (vision_mpp::BuildVisionRequest(moved, &request)) {
        std::cerr << "FAIL: empty MPP frame produced a request\n";
        return 1;
    }
    VisionServiceProfile profile;
    profile.components.push_back(
        {"image_preprocess.opencl", 1.0, 1});
    if (vision_mpp::FindImagePreprocessBackend(profile) !=
        "opencl") {
        std::cerr
            << "FAIL: OpenCL preprocess profile was not recognized\n";
        return 1;
    }
    profile.components[0].name = "image_preprocess.cpu";
    if (vision_mpp::FindImagePreprocessBackend(profile) != "cpu") {
        std::cerr
            << "FAIL: CPU preprocess profile was not recognized\n";
        return 1;
    }
    moved.reset();

    std::cout << "PASS: MPP frame source API\n";
    return 0;
}
