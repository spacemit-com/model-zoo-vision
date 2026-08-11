/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <type_traits>
#include <utility>

#include "mpp_example_helpers.h"

int main()
{
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
