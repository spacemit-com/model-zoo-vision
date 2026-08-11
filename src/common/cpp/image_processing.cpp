/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_processing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)

namespace fs = std::filesystem;

namespace vision_common {

cv::Mat letterbox(
    const cv::Mat& image,
    const std::pair<int, int>& target_size,
    const cv::Scalar& pad_color) {
    int input_h = target_size.first;   // height
    int input_w = target_size.second;  // width

    int orig_h = image.rows;
    int orig_w = image.cols;

    // Calculate scale ratio (similar to Python: min(input_h/orig_h, input_w/orig_w))
    float r = std::min(static_cast<float>(input_h) / orig_h,
                        static_cast<float>(input_w) / orig_w);

    // Compute new dimensions
    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));

    // Compute padding (similar to Python: dw, dh = (input_w - new_w)/2, (input_h - new_h)/2)
    float dw = (input_w - new_w) / 2.0f;
    float dh = (input_h - new_h) / 2.0f;

    // Resize image if needed
    cv::Mat resized;
    if (orig_w != new_w || orig_h != new_h) {
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    // Add padding (similar to Python: top, bottom = round(dh-0.1), round(dh+0.1))
    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                        cv::BORDER_CONSTANT, pad_color);

    return padded;
}

cv::Mat letterbox_to_nchw_rgb_blob(
    const cv::Mat& image,
    const std::pair<int, int>& target_size,
    const cv::Scalar& pad_color,
    cv::Mat* resized_scratch) {
    const int input_height = target_size.first;
    const int input_width = target_size.second;
    if (image.empty()) {
        throw std::invalid_argument(
            "letterbox_to_nchw_rgb_blob: input image is empty");
    }
    if (image.type() != CV_8UC3) {
        throw std::invalid_argument(
            "letterbox_to_nchw_rgb_blob: expected CV_8UC3 BGR input");
    }
    if (input_width <= 0 || input_height <= 0) {
        throw std::invalid_argument(
            "letterbox_to_nchw_rgb_blob: target size must be positive");
    }
    const float ratio = std::min(
        static_cast<float>(input_height) / image.rows,
        static_cast<float>(input_width) / image.cols);
    const int resized_width = static_cast<int>(
        std::round(image.cols * ratio));
    const int resized_height = static_cast<int>(
        std::round(image.rows * ratio));
    const float dw = (input_width - resized_width) / 2.0F;
    const float dh = (input_height - resized_height) / 2.0F;
    const int left = static_cast<int>(std::round(dw - 0.1F));
    const int right = static_cast<int>(std::round(dw + 0.1F));
    const int top = static_cast<int>(std::round(dh - 0.1F));
    const int bottom = static_cast<int>(std::round(dh + 0.1F));

    thread_local cv::Mat thread_resized_scratch;
    cv::Mat* resize_output =
        resized_scratch != nullptr
            ? resized_scratch
            : &thread_resized_scratch;
    const cv::Mat* resized = &image;
    if (image.cols != resized_width || image.rows != resized_height) {
        cv::resize(
            image,
            *resize_output,
            cv::Size(resized_width, resized_height),
            0.0,
            0.0,
            cv::INTER_LINEAR);
        resized = resize_output;
    }

    const int dimensions[] = {
        1, 3, input_height, input_width};
    cv::Mat output_blob(4, dimensions, CV_32F);
    float* output = output_blob.ptr<float>();
    const size_t plane_size =
        static_cast<size_t>(input_width) * input_height;
    constexpr float scale = 1.0F / 255.0F;
    const float channel_padding[] = {
        static_cast<float>(pad_color[2]) * scale,
        static_cast<float>(pad_color[1]) * scale,
        static_cast<float>(pad_color[0]) * scale};

    for (int channel = 0; channel < 3; ++channel) {
        float* plane = output +
            static_cast<size_t>(channel) * plane_size;
        const float padding = channel_padding[channel];
        if (top > 0) {
            std::fill(
                plane,
                plane + static_cast<size_t>(top) * input_width,
                padding);
        }
        if (bottom > 0) {
            std::fill(
                plane + static_cast<size_t>(top + resized_height) *
                    input_width,
                plane + plane_size,
                padding);
        }
        if (left > 0 || right > 0) {
            for (int y = 0; y < resized_height; ++y) {
                float* row = plane +
                    static_cast<size_t>(top + y) * input_width;
                std::fill(row, row + left, padding);
                std::fill(
                    row + left + resized_width,
                    row + input_width,
                    padding);
            }
        }
    }

    cv::parallel_for_(
        cv::Range(0, resized_height),
        [&](const cv::Range& rows) {
            float* red = output;
            float* green = output + plane_size;
            float* blue = output + plane_size * 2;
            for (int y = rows.start; y < rows.end; ++y) {
                const uint8_t* source = resized->ptr<uint8_t>(y);
                const size_t offset =
                    static_cast<size_t>(top + y) * input_width +
                    left;
                float* output_red = red + offset;
                float* output_green = green + offset;
                float* output_blue = blue + offset;
                for (int x = 0; x < resized_width; ++x) {
                    output_blue[x] = source[x * 3] * scale;
                    output_green[x] = source[x * 3 + 1] * scale;
                    output_red[x] = source[x * 3 + 2] * scale;
                }
            }
        });
    return output_blob;
}

cv::Mat preprocess_classification(
    const cv::Mat& image,
    const std::pair<int, int>& input_shape,
    const cv::Scalar& mean,
    const cv::Scalar& std,
    const cv::Size& resize_size,
    bool center_crop,
    int interpolation) {
    cv::Mat img = image;

    // Resize on uint8 first — much faster than resizing float32 data
    if (resize_size.width > 0 && resize_size.height > 0) {
        cv::resize(img, img, resize_size, 0, 0, interpolation);
    }

    // Center crop on uint8
    if (center_crop && (resize_size.width > 0 && resize_size.height > 0)) {
        int y0 = (img.rows - input_shape.first) / 2;
        int x0 = (img.cols - input_shape.second) / 2;
        img = img(cv::Rect(x0, y0, input_shape.second, input_shape.first)).clone();
    } else {
        cv::resize(img, img, cv::Size(input_shape.second, input_shape.first),
                    0, 0, interpolation);
    }

    // blobFromImage: BGR->RGB (swapRB), float conversion, 1/255 scale,
    // and HWC->CHW in one optimized call
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0,
                                            cv::Size(), cv::Scalar(),
                                            true, false, CV_32F);

    // Per-channel mean/std normalization directly on contiguous CHW memory
    const int channel_size = input_shape.first * input_shape.second;
    float* blob_data = blob.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        float* ch = blob_data + c * channel_size;
        const float m = static_cast<float>(mean[c]) / 255.0f;
        const float inv_s = 255.0f / static_cast<float>(std[c]);
        for (int i = 0; i < channel_size; ++i) {
            ch[i] = (ch[i] - m) * inv_s;
        }
    }

    return blob;
}

std::vector<std::string> load_labels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);

    if (!file.is_open()) {
        return labels;  // Return empty vector if file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace from line
        if (!line.empty()) {
            // Remove trailing whitespace
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) {
                labels.push_back(line);
            }
        }
    }

    file.close();
    return labels;
}

std::string resolve_path_for_resource(const std::string& path) {
    if (path.empty() || (path.size() >= 1 && path[0] == '/') ||
        (path.size() >= 2 && path[1] == ':')) {
        return path;
    }
    if (fs::exists(path)) return path;
    std::string with_parent = "../" + path;
    if (fs::exists(with_parent)) return with_parent;
    return path;
}

std::vector<std::string> load_labels_imagenet(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) continue;
            // ImageNet format: "n01440764 tench, Tinca tinca" -> "tench, Tinca tinca"
            size_t pos = line.find(' ');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                labels.push_back(line.substr(pos + 1));
            } else {
                labels.push_back(line);
            }
        }
    }
    return labels;
}

}  // namespace vision_common
