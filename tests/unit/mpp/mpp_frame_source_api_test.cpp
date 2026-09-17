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
    const std::string camera =
        "input:\n  type: camera\n  source: /dev/video2\n  backend: mpp\n"
        "  width: 1280\n  height: 720\n  fps: 30\n";
    auto input = parse("test_image: old.jpg\n");
    require(!input.use_camera && input.image_path.empty() && input.width == 0);
    input = parse(camera);
    require(input.use_camera && input.camera.use_mpp &&
        input.camera.v4l2_dev == "/dev/video2" && input.camera.width == 1280 &&
        input.camera.height == 720 && input.camera.fps == 30);
    input = parse(camera, {"--image", "cli.jpg"});
    require(!input.use_camera && !input.camera.use_mpp && input.image_path == "cli.jpg");
    input = parse(camera, {"--use-camera", "--camera-id", "3", "--mpp-width", "640"});
    require(input.camera.v4l2_dev.empty() && input.camera.camera_id == 3 &&
        input.width == 640 && input.height == 720 && input.camera.use_mpp);
    input = parse(camera, {"--v4l2-dev", "/dev/video4", "--camera-id", "1"});
    require(input.camera.v4l2_dev == "/dev/video4");
    input = parse("input: {type: image, source: sample.jpg}\n");
    require(input.image_path == "/tmp/sample.jpg" && !input.use_camera);
    input = parse("input: {type: image, source: sample.jpg}\n", {"--use-camera"});
    require(input.use_camera && !input.camera.use_mpp && input.camera.v4l2_dev.empty());
    input = parse("{}", {"--use-camera", "--use-mpp", "--mpp-height", "480"});
    require(input.use_camera && input.camera.use_mpp && input.camera.height == 480);
    input = parse("input: {type: camera, backend: opencv, width: 640}\n");
    require(input.use_camera && !input.camera.use_mpp && input.width == 640 && !input.fps);
    rejects("input: camera\n");
    rejects("input: {type: video, source: movie.mp4}\n");
    rejects("input: {type: image, backend: mpp}\n");
    rejects("input: {type: camera, backend: invalid}\n");
    rejects("input: {type: camera, source: 'rtsp://example'}\n");
    rejects("input: {type: camera, width: 0}\n");
    rejects("input: {type: image, width: 640}\n");
    rejects(camera, {"--image", "a.jpg", "--use-camera"});
    rejects(camera, {"--use-camera", "--image", "a.jpg"});
    rejects(camera, {"--camera-id", "-1"});
    rejects(camera, {"--mpp-width", "640oops"});
    rejects(camera, {"--mpp-fps"});
    rejects("{}", {"--use-mpp"});
    rejects("{}", {"--no-display"});
    rejects("{}", {"--video", "a.mp4"});

    // Existing application/tracking parsers retain their business arguments
    // and pass only resolved input overrides to the common configuration code.
    auto legacy = [&](const std::string& yaml, bool camera_requested,
                        bool id_set, const std::string& media, bool video,
                        bool mpp_supported) {
        std::ofstream config(file.path);
        config << yaml;
        config.close();
        return vision_mpp::ConfigureLegacyDemoInput(
            file.path, camera_requested, 3, id_set, media, video,
            mpp_supported, &input);
    };
    require(legacy("{}", false, false, "", true, true));
    require(!input.use_camera && input.image_path.empty());
    require(legacy(camera, false, false, "", true, true));
    require(input.use_camera && input.camera.use_mpp && input.width == 1280);
    require(legacy(camera, false, true, "", true, true));
    require(input.camera.v4l2_dev.empty() && input.camera.camera_id == 3);
    require(legacy(camera, false, true, "clip.mp4", true, true));
    require(!input.use_camera && !input.camera.use_mpp);
    require(!legacy(camera, false, false, "", false, false));
    require(legacy(camera, false, false, "photo.jpg", false, false));
    require(!input.use_camera && input.image_path == "photo.jpg");
    require(legacy("input: {type: image}\n", true, true, "", false, false));
    require(input.use_camera && !input.camera.use_mpp && input.camera.camera_id == 3);
    require(legacy("input: {type: camera, backend: opencv, width: 800}\n",
        false, false, "", false, false));
    require(input.use_camera && input.width == 800);
    require(!legacy("input: {type: image}\n", false, false, "", true, true));
    require(legacy("input: {type: image}\n", false, false, "clip.mp4", true, true));
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
