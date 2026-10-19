/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_seg_detector.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

// Resolve the single-output seg layout for output0 dims [dim1, dim2].
// cn layout = [features, anchors] (Ultralytics default); features sit on the smaller
// trailing dim (channels << anchors). Shared by load_model() and the postprocess path
// so both agree on which axis is features vs anchors. Equal dims are ambiguous.
struct SegLayout {
    bool cn;            // true if [features, anchors]
    int64_t features;
    int64_t anchors;
};

SegLayout resolve_seg_layout(int64_t dim1, int64_t dim2) {
    if (dim1 <= 0 || dim2 <= 0 || dim1 == dim2) {
        throw std::runtime_error(
            "YOLOv8-Seg: ambiguous output0 shape (channels == anchors), "
            "cannot distinguish features from anchors");
    }
    if (dim1 < dim2) {
        return SegLayout{true, dim1, dim2};
    }
    return SegLayout{false, dim2, dim1};
}

// Validate before reading typed data or indexing a tensor's spatial dimensions.
std::vector<int64_t> seg_tensor_shape(const Ort::Value& tensor, size_t rank) {
    if (!tensor.IsTensor()) throw std::runtime_error("YOLOv8-Seg expects tensor outputs");
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    const auto dims = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        dims.size() != rank || dims[0] != 1 ||
        !std::all_of(dims.begin(), dims.end(), [](int64_t d) {
            return d > 0 && d <= std::numeric_limits<int>::max();
        })) {
        throw std::runtime_error("YOLOv8-Seg expects float32 outputs with batch 1 and positive dimensions");
    }
    return dims;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOv8SegDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8SegDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8SegDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8SegDetector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv8SegDetector::YOLOv8SegDetector(const std::string& model_path,
                                        float conf_threshold,
                                        float iou_threshold,
                                        int num_threads,
                                        bool lazy_load,
                                        const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        num_classes_(0),
        proto_channels_(0),
        provider_(provider) {
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8SegDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] != 1 || input_shape_[1] != 3 ||
        input_shape_[2] <= 0 || input_shape_[3] <= 0 ||
        input_shape_[2] > std::numeric_limits<int>::max() ||
        input_shape_[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("YOLOv8-Seg expects input [1,3,H,W] with fixed positive H,W");
    }
    if (output_num_ != 2 && output_num_ < 13) {
        throw std::runtime_error("YOLOv8-Seg expects 2 or 13 outputs");
    }

    if (output_num_ == 2) {
        // Ultralytics export: output0 = [1, 4 + num_classes + proto_channels, num_anchors],
        // proto = [1, proto_channels, mask_h, mask_w].
        Ort::TypeInfo proto_type_info = session_->GetOutputTypeInfo(1);
        auto proto_dims = proto_type_info.GetTensorTypeAndShapeInfo().GetShape();
        // Expect 4-D [1, C, H, W]; reject e.g. a batch-less [C, H, W] where dims[1]
        // would be a spatial size rather than the channel count.
        if (proto_dims.size() != 4) {
            throw std::runtime_error(
                "YOLOv8-Seg: unexpected proto output rank (expected 4: [1, C, H, W])");
        }
        if (proto_dims[1] > 0) {
            proto_channels_ = static_cast<int>(proto_dims[1]);
        }
        if (proto_channels_ <= 0) {
            throw std::runtime_error(
                "YOLOv8-Seg: cannot read proto_channels from proto output shape");
        }

        Ort::TypeInfo out0_type_info = session_->GetOutputTypeInfo(0);
        auto out0_dims = out0_type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (out0_dims.size() != 3) {
            throw std::runtime_error("YOLOv8-Seg: unexpected output0 rank (expected 3)");
        }
        // Shared layout resolver throws on ambiguous (features == anchors) shapes rather
        // than silently mis-deriving num_classes and shifting the mask-coefficient channels.
        const SegLayout layout = resolve_seg_layout(out0_dims[1], out0_dims[2]);
        const int derived = static_cast<int>(layout.features) - 4 - proto_channels_;
        if (derived <= 0) {
            throw std::runtime_error(
                "YOLOv8-Seg: derived num_classes <= 0 from output0 shape");
        }
        num_classes_ = derived;
    } else {
        // DFL multi-branch export:
        // [box0, score0, sum0, box1, score1, sum1, box2, score2, sum2, seg0, seg1, seg2, proto]
        // score output shape: [1, num_classes, h, w]
        if (output_num_ >= 2) {
            Ort::TypeInfo score_type_info = session_->GetOutputTypeInfo(1);
            auto score_dims = score_type_info.GetTensorTypeAndShapeInfo().GetShape();
            if (score_dims.size() >= 2 && score_dims[1] > 0) {
                num_classes_ = static_cast<int>(score_dims[1]);
            }
        }
        // proto output shape: [1, proto_channels, mask_h, mask_w]
        if (output_num_ >= 13) {
            Ort::TypeInfo proto_type_info = session_->GetOutputTypeInfo(12);
            auto proto_dims = proto_type_info.GetTensorTypeAndShapeInfo().GetShape();
            if (proto_dims.size() >= 2 && proto_dims[1] > 0) {
                proto_channels_ = static_cast<int>(proto_dims[1]);
            }
        }
    }

    // Default values if not found
    if (num_classes_ <= 0) {
        num_classes_ = 80;  // Default COCO classes
    }
    if (proto_channels_ <= 0) {
        proto_channels_ = 32;  // Default proto channels
    }

    model_loaded_ = true;
}

cv::Mat YOLOv8SegDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    return vision_common::letterbox_to_nchw_rgb_blob(
        image,
        std::make_pair(inputHeight, inputWidth));
}

vision_common::SegmentationResultList YOLOv8SegDetector::segment(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    vision_core::ImageInput input;
    input.image = image;
    return segment_input(input, conf_threshold, iou_threshold);
}

vision_common::SegmentationResultList
YOLOv8SegDetector::segment_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold, int max_det, float mask_threshold) {
    if (!std::isfinite(mask_threshold) || mask_threshold > 1.0f) {
        throw std::invalid_argument("mask_threshold must be finite and <= 1");
    }
    const float use_mask = mask_threshold > 0.0f ? mask_threshold : 0.5f;
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    const cv::Size orig_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows);

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = static_cast<int>(input_shape_[3]);
    spec.output_height = static_cast<int>(input_shape_[2]);
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    spec.padding = {114.0F, 114.0F, 114.0F};
    auto prepared = prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::SegmentationResultList results = postprocess_impl(
        outputs, orig_size, use_conf, use_iou, max_det, use_mask);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> YOLOv8SegDetector::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

vision_core::InferResponse YOLOv8SegDetector::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kSegment) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv8SegDetector expects ImageInput";
        return response;
    }

    vision_common::SegmentationResultList task_results =
        segment_input(
            *image_input,
            request.params.conf_threshold,
            request.params.iou_threshold, request.params.max_det,
            request.params.mask_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> YOLOv8SegDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::SegmentationResultList YOLOv8SegDetector::postprocess(std::vector<Ort::Value>& outputs,
                                                                const cv::Size& orig_size,
                                                                float conf_threshold,
                                                                float iou_threshold) {
    return postprocess_impl(outputs, orig_size, conf_threshold, iou_threshold, -1, 0.5f);
}

vision_common::SegmentationResultList YOLOv8SegDetector::postprocess_impl(
    std::vector<Ort::Value>& outputs, const cv::Size& orig_size,
    float conf_threshold, float iou_threshold, int max_det, float mask_threshold) {
    vision_common::mask_logit_threshold(mask_threshold);
    if (orig_size.width <= 0 || orig_size.height <= 0) {
        throw std::invalid_argument("YOLOv8-Seg expects a positive original image size");
    }
    // outputs.size() == output_num_ (session output node count, fixed by the model graph):
    // 2 outputs -> Ultralytics single-output seg export; otherwise -> multi-branch DFL heads.
    // Both paths must agree with load_model(), which keys num_classes_/proto_channels_ off
    // the same output_num_.
    if (outputs.size() == 2) {
        return postprocess_single_output_seg(
            outputs, orig_size, conf_threshold, iou_threshold, max_det, mask_threshold);
    }

    // DFL path indexes outputs[0..8] (box/score/score_sum x3), outputs[9..11] (seg coeffs)
    // and outputs[12] (proto), so it requires the full 13-output export. Guard the lower
    // bound up front to fail clearly instead of reading out of bounds on a malformed model.
    if (outputs.size() < 13) {
        throw std::runtime_error(
            "YOLOv8-Seg: expected 2 (single-output) or 13 (DFL multi-branch) outputs, got "
            + std::to_string(outputs.size()));
    }

    const auto proto_shape = seg_tensor_shape(outputs[12], 4);
    if (proto_shape[1] != proto_channels_) {
        throw std::runtime_error("YOLOv8-Seg proto channels changed");
    }
    for (int branch = 0; branch < 3; ++branch) {
        const auto box = seg_tensor_shape(outputs[branch * 3], 4);
        const auto scores = seg_tensor_shape(outputs[branch * 3 + 1], 4);
        const auto sums = seg_tensor_shape(outputs[branch * 3 + 2], 4);
        const auto coeffs = seg_tensor_shape(outputs[9 + branch], 4);
        if (box[1] != 64 || scores[1] != num_classes_ || sums[1] != 1 ||
            coeffs[1] != proto_channels_ ||
            box[2] * box[3] > std::numeric_limits<int>::max() ||
            scores[2] != box[2] || scores[3] != box[3] ||
            sums[2] != box[2] || sums[3] != box[3] ||
            coeffs[2] != box[2] || coeffs[3] != box[3]) {
            throw std::runtime_error("YOLOv8-Seg DFL head channels or grids do not match");
        }
    }
    if (max_det == 0) return {};

    // Temporary structure to hold results with mask coefficients before NMS
    struct TempSegResult {
        vision_common::SegmentationResult result;
        std::vector<float> mask_coeffs;
    };

    std::vector<TempSegResult> temp_objects;

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    // Process 3 branches (8x, 16x, 32x downsampling)
    for (int i = 0; i < 3; i++) {
        const float* boxes = outputs[i * 3].GetTensorMutableData<float>();
        const float* scores = outputs[i * 3 + 1].GetTensorMutableData<float>();
        const float* score_sum = outputs[i * 3 + 2].GetTensorMutableData<float>();
        const float* seg_part = outputs[9 + i].GetTensorMutableData<float>();
        std::vector<int64_t> dims = outputs[i * 3].GetTensorTypeAndShapeInfo().GetShape();

        int grid_h = static_cast<int>(dims[2]);
        int grid_w = static_cast<int>(dims[3]);
        int anchors_per_branch = grid_w * grid_h;
        float scale_w = static_cast<float>(inputWidth) / static_cast<float>(grid_w);
        float scale_h = static_cast<float>(inputHeight) / static_cast<float>(grid_h);

        int orig_height = orig_size.height;
        int orig_width = orig_size.width;
        float scale2orign = std::min(
            static_cast<float>(inputHeight) / static_cast<float>(orig_height),
            static_cast<float>(inputWidth) / static_cast<float>(orig_width));
        int pad_h = static_cast<int>(std::round(
            (inputHeight - std::round(orig_height * scale2orign)) / 2.0f - 0.1f));
        int pad_w = static_cast<int>(std::round(
            (inputWidth - std::round(orig_width * scale2orign)) / 2.0f - 0.1f));

        for (int anchor_idx = 0; anchor_idx < anchors_per_branch; anchor_idx++) {
            if (score_sum[anchor_idx] < conf_threshold) {
                continue;
            }

            float max_score = -1.0f;
            int classId = -1;
            for (int class_idx = 0; class_idx < num_classes_; class_idx++) {
                size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
                if (std::isfinite(scores[score_offset]) &&
                    scores[score_offset] > conf_threshold && scores[score_offset] > max_score) {
                    max_score = scores[score_offset];
                    classId = class_idx;
                }
            }

            if (classId >= 0) {
                auto [x1, y1, x2, y2] = vision_common::dfl_decode(boxes, anchor_idx, anchors_per_branch,
                    grid_w, scale_w, scale_h, scale2orign, pad_w, pad_h);

                if (!std::isfinite(x1) || !std::isfinite(y1) ||
                    !std::isfinite(x2) || !std::isfinite(y2) || x2 <= x1 || y2 <= y1) continue;
                // Reject empty image intersections before capping, but keep the
                // original decoded boxes for NMS so clipping does not change IoU.
                if (std::min(x2, static_cast<float>(orig_size.width)) <= std::max(x1, 0.0f) ||
                    std::min(y2, static_cast<float>(orig_size.height)) <= std::max(y1, 0.0f)) continue;
                TempSegResult temp;
                temp.result.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
                temp.result.label = classId;
                temp.result.score = max_score;
                temp.result.mask = nullptr;

                // Store mask coefficients
                temp.mask_coeffs.reserve(proto_channels_);
                for (int k = 0; k < proto_channels_; k++) {
                    temp.mask_coeffs.push_back(seg_part[k * anchors_per_branch + anchor_idx]);
                }

                if (!std::all_of(temp.mask_coeffs.begin(), temp.mask_coeffs.end(),
                    [](float value) { return std::isfinite(value); })) continue;
                temp_objects.push_back(std::move(temp));
            }
        }
    }

    // Extract just the results for NMS
    vision_common::SegmentationResultList objects;
    objects.reserve(temp_objects.size());
    for (const auto& temp : temp_objects) {
        objects.push_back(temp.result);
    }

    // Keep candidate identity through NMS instead of matching floating-point
    // boxes/scores back to coefficients. Limit results before mask assembly.
    auto kept = vision_common::multi_class_nms_indices(objects, iou_threshold);
    vision_common::limit_by_score(kept, max_det,
        [&](size_t index) { return objects[index].score; });
    vision_common::SegmentationResultList results;
    std::vector<std::vector<float>> mask_coeffs_list;
    results.reserve(kept.size());
    mask_coeffs_list.reserve(kept.size());
    for (size_t index : kept) {
        auto& box = temp_objects[index].result.bbox;
        box.x1 = std::clamp(box.x1, 0.0f, static_cast<float>(orig_size.width));
        box.x2 = std::clamp(box.x2, 0.0f, static_cast<float>(orig_size.width));
        box.y1 = std::clamp(box.y1, 0.0f, static_cast<float>(orig_size.height));
        box.y2 = std::clamp(box.y2, 0.0f, static_cast<float>(orig_size.height));
        results.push_back(std::move(temp_objects[index].result));
        mask_coeffs_list.push_back(std::move(temp_objects[index].mask_coeffs));
    }
    if (!results.empty()) {
        const float* output_proto = outputs[12].GetTensorMutableData<float>();
        const auto proto_dims = outputs[12].GetTensorTypeAndShapeInfo().GetShape();
        auto masks = _process_masks(
            output_proto, proto_dims, mask_coeffs_list, results, orig_size, mask_threshold);
        for (size_t i = 0; i < results.size(); ++i) results[i].mask = std::move(masks[i]);
    }

    return results;
}


vision_common::SegmentationResultList YOLOv8SegDetector::postprocess_single_output_seg(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold, int max_det, float mask_threshold) {
    Ort::Value& out0 = outputs[0];
    std::vector<int64_t> dims = seg_tensor_shape(out0, 3);
    if (dims.size() != 3) {
        throw std::runtime_error("Unexpected YOLOv8-Seg single output rank (expected 3)");
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);

    std::vector<int64_t> proto_dims = seg_tensor_shape(outputs[1], 4);
    // Expect 4-D [1, C, H, W] (same assumption as load_model); reject a batch-less
    // [C, H, W] where dims[1] would be a spatial size rather than the channel count.
    if (proto_dims.size() != 4 || proto_dims[1] <= 0) {
        throw std::runtime_error(
            "YOLOv8-Seg proto output shape invalid (expected 4: [1, C, H, W])");
    }
    const int num_masks = static_cast<int>(proto_dims[1]);

    // Same resolver as load_model() so feature/anchor axes are chosen identically.
    const SegLayout layout = resolve_seg_layout(dims[1], dims[2]);
    const bool layout_cn = layout.cn;
    const int64_t features = layout.features;
    const int64_t num_anchors = layout.anchors;

    const int num_classes = static_cast<int>(features) - 4 - num_masks;
    if (num_classes <= 0) {
        throw std::runtime_error("YOLOv8-Seg single output feature size too small");
    }

    if (max_det == 0) return {};
    const float* data = out0.GetTensorData<float>();
    auto at = [&](int c, int64_t i) -> float {
        return layout_cn ? data[static_cast<size_t>(c) * num_anchors + i]
            : data[static_cast<size_t>(i) * features + c];
    };

    // Collect candidates: max class score over [4, 4+num_classes), keep mask coeffs aligned.
    // Ultralytics exports box xywh in input-pixel scale, so no normalization rescale.
    std::vector<cv::Rect2f> cand_rects;   // xywh in input space, for NMS
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;
    std::vector<std::vector<float>> cand_coeffs;
    cand_rects.reserve(256);
    cand_scores.reserve(256);
    cand_labels.reserve(256);
    cand_coeffs.reserve(256);

    const float gain = std::min(
        static_cast<float>(input_height) / static_cast<float>(orig_size.height),
        static_cast<float>(input_width) / static_cast<float>(orig_size.width));
    const float pad_w = std::round((input_width - std::round(orig_size.width * gain)) / 2.0f - 0.1f);
    const float pad_h = std::round((input_height - std::round(orig_size.height * gain)) / 2.0f - 0.1f);
    const float max_x = static_cast<float>(orig_size.width);
    const float max_y = static_cast<float>(orig_size.height);


    for (int64_t i = 0; i < num_anchors; ++i) {
        float best = -1.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            const float s = at(4 + c, i);
            if (std::isfinite(s) && s > best) {
                best = s;
                best_cls = c;
            }
        }
        if (best_cls < 0 || best < conf_threshold) {
            continue;
        }

        const float w = at(2, i);
        const float h = at(3, i);
        const float x = at(0, i) - w * 0.5f;
        const float y = at(1, i) - h * 0.5f;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) ||
            !std::isfinite(h) || !std::isfinite(x + w) || !std::isfinite(y + h) ||
            w <= 0.0f || h <= 0.0f) continue;
        const float x1 = std::clamp((x - pad_w) / gain, 0.0f, max_x);
        const float y1 = std::clamp((y - pad_h) / gain, 0.0f, max_y);
        const float x2 = std::clamp((x + w - pad_w) / gain, 0.0f, max_x);
        const float y2 = std::clamp((y + h - pad_h) / gain, 0.0f, max_y);
        if (x2 <= x1 || y2 <= y1) continue;

        std::vector<float> coeffs;
        coeffs.reserve(num_masks);
        for (int k = 0; k < num_masks; ++k) {
            coeffs.push_back(at(4 + num_classes + k, i));
        }
        if (!std::all_of(coeffs.begin(), coeffs.end(),
            [](float value) { return std::isfinite(value); })) continue;
        cand_rects.emplace_back(x, y, w, h);
        cand_scores.push_back(best);
        cand_labels.push_back(best_cls);
        cand_coeffs.push_back(std::move(coeffs));
    }

    if (cand_rects.empty()) {
        return {};
    }

    std::vector<std::vector<size_t>> by_class(static_cast<size_t>(num_classes));
    for (size_t i = 0; i < cand_rects.size(); ++i) {
        by_class[static_cast<size_t>(cand_labels[i])].push_back(i);
    }

    std::vector<size_t> kept_candidates;
    for (const auto& cls_indices : by_class) {
        if (cls_indices.empty()) continue;
        std::vector<cv::Rect2f> rects;
        std::vector<float> scores;
        rects.reserve(cls_indices.size());
        scores.reserve(cls_indices.size());
        for (size_t idx : cls_indices) {
            rects.push_back(cand_rects[idx]);
            scores.push_back(cand_scores[idx]);
        }
        for (int index : vision_common::nms(rects, scores, iou_threshold)) {
            kept_candidates.push_back(cls_indices[static_cast<size_t>(index)]);
        }
    }
    vision_common::limit_by_score(kept_candidates, max_det,
        [&](size_t index) { return cand_scores[index]; });
    vision_common::SegmentationResultList results;
    std::vector<std::vector<float>> kept_coeffs;
    results.reserve(kept_candidates.size());
    kept_coeffs.reserve(kept_candidates.size());
    for (size_t idx : kept_candidates) {
        const cv::Rect2f& b = cand_rects[idx];
        float x1 = (b.x - pad_w) / gain;
        float y1 = (b.y - pad_h) / gain;
        float x2 = (b.x + b.width - pad_w) / gain;
        float y2 = (b.y + b.height - pad_h) / gain;
        x1 = std::max(0.0f, std::min(x1, max_x));
        y1 = std::max(0.0f, std::min(y1, max_y));
        x2 = std::max(0.0f, std::min(x2, max_x));
        y2 = std::max(0.0f, std::min(y2, max_y));
        vision_common::SegmentationResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.label = cand_labels[idx];
        r.score = cand_scores[idx];
        r.mask = nullptr;
        results.push_back(std::move(r));
        kept_coeffs.push_back(std::move(cand_coeffs[idx]));
    }

    if (results.empty()) {
        return results;
    }

    // proto = outputs[1] = [1, proto_channels, mask_h, mask_w]
    const float* proto = outputs[1].GetTensorData<float>();
    std::vector<std::shared_ptr<cv::Mat>> masks =
        _process_masks(proto, proto_dims, kept_coeffs, results, orig_size, mask_threshold);
    for (size_t i = 0; i < results.size() && i < masks.size(); ++i) {
        results[i].mask = masks[i];
    }

    return results;
}


std::vector<std::shared_ptr<cv::Mat>> YOLOv8SegDetector::_process_masks(
    const float* protos,
    const std::vector<int64_t>& proto_dims,
    const std::vector<std::vector<float>>& mask_coeffs,
    const vision_common::SegmentationResultList& results,
    const cv::Size& orig_shape, float mask_threshold) {
    if (proto_dims.size() != 4 || proto_dims[0] != 1 ||
        proto_dims[1] <= 0 || proto_dims[1] > std::numeric_limits<int>::max() ||
        proto_dims[2] <= 0 || proto_dims[3] <= 0 ||
        proto_dims[2] > std::numeric_limits<int>::max() / proto_dims[3] ||
        mask_coeffs.size() != results.size()) {
        throw std::invalid_argument("Invalid instance mask dimensions or candidate count");
    }
    for (const auto& coefficients : mask_coeffs) {
        if (coefficients.size() != static_cast<size_t>(proto_dims[1])) {
            throw std::invalid_argument("Instance mask coefficient count does not match prototype channels");
        }
    }
    if (results.empty() || mask_coeffs.empty()) {
        return std::vector<std::shared_ptr<cv::Mat>>();
    }

    int mask_dim = static_cast<int>(proto_dims[1]);
    int mask_h = static_cast<int>(proto_dims[2]);
    int mask_w = static_cast<int>(proto_dims[3]);
    int proto_size = mask_h * mask_w;
    int num_masks = static_cast<int>(mask_coeffs.size());

    // Mask assembly is a GEMM: raw_masks[num_masks, proto_size] =
    //   mask_coeffs[num_masks, mask_dim] * protos[mask_dim, proto_size].
    // Expressed via Eigen; EIGEN_USE_BLAS (CMake) routes GEMM to OpenBLAS.
    std::vector<float> raw_masks(static_cast<size_t>(num_masks) * proto_size, 0.0f);

    // Pack mask coefficients into a contiguous row-major [num_masks, mask_dim] matrix.
    using RowMatrixXf =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMatrixXf coeff_mat(num_masks, mask_dim);
    for (int i = 0; i < num_masks; ++i) {
        for (int k = 0; k < mask_dim; ++k) {
            coeff_mat(i, k) = mask_coeffs[i][k];
        }
    }
    Eigen::Map<const RowMatrixXf> proto_mat(protos, mask_dim, proto_size);
    Eigen::Map<RowMatrixXf> raw_mat(raw_masks.data(), num_masks, proto_size);
    raw_mat.noalias() = coeff_mat * proto_mat;
    std::vector<std::shared_ptr<cv::Mat>> masks;
    masks.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        cv::Mat logits(mask_h, mask_w, CV_32F, raw_masks.data() + i * proto_size);
        masks.push_back(std::make_shared<cv::Mat>(vision_common::restore_instance_mask(
            logits, results[i].bbox, orig_shape, mask_threshold)));
    }
    return masks;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8SegDetector> registrar("YOLOv8SegDetector");

}  // namespace vision_deploy
