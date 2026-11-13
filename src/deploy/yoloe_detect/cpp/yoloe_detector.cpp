/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "yoloe_detector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cpp/embedding_utils.h"
#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {
constexpr int kSlots = 80;
constexpr int kEmbedding = 512;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
vision_operators::ImagePreprocessSpec image_spec(int side) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = side;
    spec.output_height = side;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kLetterbox;
    spec.padding = {114.F, 114.F, 114.F};
    return spec;
}
cv::Mat pixels(const cv::Mat& image, int side, vision_operators::ImagePreprocessGeometry* geometry) {
    vision_operators::CpuChannelTransform transform;
    transform.input_divisor = {255.F, 255.F, 255.F};
    return vision_operators::preprocess_bgr_to_nchw(image, image_spec(side), transform, geometry);
}
void require_tensor(Ort::Session& session, bool input, size_t index, const std::vector<int64_t>& shape,
    ONNXTensorElementDataType type) {
    const auto description = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const auto info = description.GetTensorTypeAndShapeInfo();
    const auto actual = info.GetShape();
    if (info.GetElementType() != type || actual.size() != shape.size()) {
        throw std::runtime_error("YOLOE detection tensor dtype/rank mismatch");
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] >= 0 && shape[i] != actual[i]) {
            throw std::runtime_error("YOLOE detection tensor shape mismatch");
        }
    }
}
void validate_names(const std::vector<std::string>& names) {
    if (names.empty() || names.size() > kSlots) {
        throw std::invalid_argument("YOLOE requires 1..80 class names");
    }
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].find_first_not_of(" \t\r\n") == std::string::npos ||
            std::find(names.begin(), names.begin() + i, names[i]) != names.begin() + i) {
            throw std::invalid_argument("YOLOE class names must be nonempty and unique");
        }
    }
}
bool same_visuals(const std::vector<vision::VisualPrompt>& a, const std::vector<vision::VisualPrompt>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].class_name != b[i].class_name || a[i].bboxes.size() != b[i].bboxes.size() ||
            a[i].image.size() != b[i].image.size() || a[i].image.type() != b[i].image.type())
            return false;
        for (size_t j = 0; j < a[i].bboxes.size(); ++j) {
            const auto& x = a[i].bboxes[j];
            const auto& y = b[i].bboxes[j];
            if (x.x1 != y.x1 || x.y1 != y.y1 || x.x2 != y.x2 || x.y2 != y.y2)
                return false;
        }
        for (int row = 0; row < a[i].image.rows; ++row) {
            if (std::memcmp(a[i].image.ptr(row), b[i].image.ptr(row), a[i].image.cols * a[i].image.elemSize()) != 0)
                return false;
        }
    }
    return true;
}
} // namespace

std::vector<std::string> yoloe_prompt_names(const std::vector<std::string>& texts,
    const std::vector<vision::VisualPrompt>& visuals, const std::vector<std::string>& defaults) {
    std::vector<std::string> names = texts;
    if (names.empty() && visuals.empty())
        names = defaults;
    for (const auto& prompt : visuals) {
        if (prompt.image.empty() || prompt.image.type() != CV_8UC3 || prompt.bboxes.empty()) {
            throw std::invalid_argument("Visual prompt requires BGR8 reference and boxes");
        }
        for (const auto& box : prompt.bboxes) {
            if (!std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) || !std::isfinite(box.y2) ||
                box.x1 < 0 || box.y1 < 0 || box.x2 > prompt.image.cols || box.y2 > prompt.image.rows ||
                box.x1 >= box.x2 || box.y1 >= box.y2) {
                throw std::invalid_argument("Visual prompt box must be finite and inside reference");
            }
        }
        const auto found = std::find(names.begin(), names.end(), prompt.class_name);
        if (texts.empty() && found == names.end())
            names.push_back(prompt.class_name);
        if (!texts.empty() && found == names.end()) {
            throw std::invalid_argument("Visual class is absent from text prompts");
        }
    }
    validate_names(names);
    if (!visuals.empty()) {
        for (const auto& name : names) {
            if (std::none_of(visuals.begin(), visuals.end(), [&](const auto& p) { return p.class_name == name; }))
                throw std::invalid_argument("Every fused class needs a visual example");
        }
    }
    return names;
}

std::vector<float> yoloe_visual_mask(const vision::VisualPrompt& prompt, int side) {
    (void)yoloe_prompt_names({}, {prompt}, {});
    if (side < 32 || side % 32 != 0) {
        throw std::invalid_argument("Invalid VPE image size");
    }
    const int grid = side / 8;
    // Use the image packer's actual rounded resize and padding on both axes.
    // A separately rounded nominal gain can shift masks by a pixel at .5 ties.
    const auto geometry =
        vision_operators::make_image_preprocess_geometry(image_spec(side), prompt.image.cols, prompt.image.rows);
    const float scale_x = geometry.dst_width / geometry.src_width;
    const float scale_y = geometry.dst_height / geometry.src_height;
    std::vector<float> mask(static_cast<size_t>(grid) * grid, 0.F);
    for (const auto& box : prompt.bboxes) {
        const float x1 = (box.x1 * scale_x + geometry.dst_x) * 0.125F;
        const float y1 = (box.y1 * scale_y + geometry.dst_y) * 0.125F;
        const float x2 = (box.x2 * scale_x + geometry.dst_x) * 0.125F;
        const float y2 = (box.y2 * scale_y + geometry.dst_y) * 0.125F;
        for (int y = 0; y < grid; ++y) {
            for (int x = 0; x < grid; ++x) {
                if (x >= x1 && x < x2 && y >= y1 && y < y2) {
                    mask[static_cast<size_t>(y) * grid + x] = 1.F;
                }
            }
        }
    }
    const bool empty = std::none_of(mask.begin(), mask.end(), [](float v) { return v > 0; });
    if (empty) {
        throw std::invalid_argument("Visual prompt is empty at the VPE mask resolution");
    }
    return mask;
}

vision_common::DetectionResultList decode_yoloe_detections(const float* data, size_t count, int slots, int anchors,
    int classes, const vision_operators::ImagePreprocessGeometry& geometry, float confidence, int max_det) {
    geometry.validate();
    if (!data || slots <= 0 || slots > 80 || anchors <= 0 || classes <= 0 || classes > slots ||
        count != static_cast<size_t>(4 + slots) * anchors || !std::isfinite(confidence) || confidence < 0 ||
        confidence > 1) {
        throw std::invalid_argument("Invalid YOLOE detection output/threshold");
    }
    if (!std::all_of(data, data + count, [](float x) { return std::isfinite(x); })) {
        throw std::runtime_error("Nonfinite YOLOE detection output");
    }
    if (max_det == 0)
        return {};
    std::vector<float> best(anchors);
    std::vector<int> order(anchors);
    std::iota(order.begin(), order.end(), 0);
    for (int a = 0; a < anchors; ++a) {
        best[a] = data[4 * static_cast<size_t>(anchors) + a];
        for (int c = 1; c < classes; ++c) {
            best[a] = std::max(best[a], data[(4 + c) * static_cast<size_t>(anchors) + a]);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return best[a] > best[b]; });
    order.resize(std::min(300, anchors));
    std::vector<int> pairs(order.size() * classes);
    std::iota(pairs.begin(), pairs.end(), 0);
    const auto score = [&](int pair) {
        return data[(4 + pair % classes) * static_cast<size_t>(anchors) + order[pair / classes]];
    };
    std::stable_sort(pairs.begin(), pairs.end(), [&](int a, int b) { return score(a) > score(b); });
    pairs.resize(std::min<size_t>(pairs.size(), std::min(300, anchors)));
    const float width = static_cast<float>(geometry.original_width);
    const float height = static_cast<float>(geometry.original_height);
    vision_common::DetectionResultList results;
    for (int pair : pairs) {
        const float conf = score(pair);
        if (!(conf > confidence))
            continue;
        const int a = order[pair / classes];
        vision_common::DetectionResult result;
        result.bbox = {std::clamp(geometry.restore_x(data[a]), 0.F, width),
            std::clamp(geometry.restore_y(data[anchors + a]), 0.F, height),
            std::clamp(geometry.restore_x(data[2 * anchors + a]), 0.F, width),
            std::clamp(geometry.restore_y(data[3 * anchors + a]), 0.F, height)};
        if (result.bbox.x1 >= result.bbox.x2 || result.bbox.y1 >= result.bbox.y2)
            continue;
        result.label = pair % classes;
        result.score = conf;
        results.push_back(result);
        if (max_det > 0 && results.size() >= static_cast<size_t>(max_det))
            break;
    }
    return results;
}

YOLOEDetector::YOLOEDetector(const YAML::Node& config, bool lazy_load)
    : BaseModel(vision_core::yaml_utils::getString(config, "model_path"), lazy_load) {
    namespace yu = vision_core::yaml_utils;
    yu::validateModelParams(config,
        {"text_model_path", "vpe_model_path", "bpe_merges_path", "prompt_num_threads", "prompts", "conf_threshold"});
    const auto p = config["default_params"];
    text_path_ = yu::getString(p, "text_model_path");
    vpe_path_ = yu::getString(p, "vpe_model_path");
    merges_path_ = yu::getString(p, "bpe_merges_path");
    threads_ = yu::getInt(p, "num_threads", 8);
    prompt_threads_ = yu::getInt(p, "prompt_num_threads", 2);
    confidence_ = yu::getFloat(p, "conf_threshold", 0.2F);
    provider_ = yu::getProvider(config);
    if (text_path_.empty() || vpe_path_.empty() || merges_path_.empty() || threads_ < 0 || prompt_threads_ <= 0 ||
        !std::isfinite(confidence_) || confidence_ < 0 || confidence_ > 1) {
        throw std::invalid_argument("Invalid YOLOE model paths/threads/threshold");
    }
    if (p && p["prompts"])
        defaults_ = p["prompts"].as<std::vector<std::string>>();
    if (!defaults_.empty())
        validate_names(defaults_);
    if (!lazy_load)
        load_model();
}
std::unique_ptr<vision_core::BaseModel> YOLOEDetector::create(const YAML::Node& config, bool lazy_load) {
    return std::make_unique<YOLOEDetector>(config, lazy_load);
}
void YOLOEDetector::load_model() {
    if (model_loaded_)
        return;
    if (!session_) {
        prepare_session_configs({{threads_, provider_}, {prompt_threads_, "CPUExecutionProvider"},
            {prompt_threads_, "CPUExecutionProvider"}});
    }
    init_session(threads_, provider_);
    if (session_->GetInputCount() != 2 || session_->GetOutputCount() != 1) {
        throw std::runtime_error("YOLOE detector requires image + class embeddings and one output");
    }
    require_tensor(*session_, true, 0, {1, 3, 640, 640}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    require_tensor(*session_, true, 1, {1, 80, 512}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    require_tensor(*session_, false, 0, {1, 84, 8400}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    model_loaded_ = true;
}
void YOLOEDetector::open_text() {
    if (text_session_)
        return;
    auto tokenizer = std::make_unique<ClipTokenizer>(merges_path_);
    Ort::SessionOptions options;
    configure_session_options(options, 1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto session = std::make_unique<Ort::Session>(vision_core::shared_ort_env(), text_path_.c_str(), options);
    if (session->GetInputCount() != 1 || session->GetOutputCount() != 1) {
        throw std::runtime_error("Invalid YOLOE text encoder");
    }
    require_tensor(*session, true, 0, {-1, 77}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    require_tensor(*session, false, 0, {-1, 512}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    tokenizer_ = std::move(tokenizer);
    text_session_ = std::move(session);
}
void YOLOEDetector::open_visual() {
    if (vpe_session_)
        return;
    Ort::SessionOptions options;
    configure_session_options(options, 2);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    auto session = std::make_unique<Ort::Session>(vision_core::shared_ort_env(), vpe_path_.c_str(), options);
    if (session->GetInputCount() != 2 || session->GetOutputCount() != 1) {
        throw std::runtime_error("Invalid YOLOE VPE encoder");
    }
    const auto type = session->GetInputTypeInfo(0);
    const auto shape = type.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[2] < 32 || shape[2] > 640 || shape[2] % 32 || shape[2] != shape[3]) {
        throw std::runtime_error("VPE requires a square 32..640 input divisible by 32");
    }
    const int side = static_cast<int>(shape[2]);
    require_tensor(*session, true, 0, {1, 3, side, side}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    require_tensor(*session, true, 1, {1, -1, side / 8, side / 8}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    require_tensor(*session, false, 0, {1, -1, 512}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    reference_side_ = side;
    vpe_session_ = std::move(session);
}
std::vector<float> YOLOEDetector::run_encoder(
    Ort::Session& session, std::vector<Ort::Value>& inputs, const std::vector<int64_t>& shape, const char* component) {
    std::vector<std::string> storage;
    for (size_t i = 0; i < inputs.size(); ++i) {
        storage.emplace_back(session.GetInputNameAllocated(i, allocator_).get());
    }
    std::vector<const char*> names;
    for (const auto& s : storage)
        names.push_back(s.c_str());
    const auto output_name = session.GetOutputNameAllocated(0, allocator_);
    const char* output = output_name.get();
    const auto start = Clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(), inputs.size(), &output, 1);
    add_runtime_component_timing(component, elapsed(start));
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || info.GetShape() != shape)
        throw std::runtime_error("Encoder output contract changed");
    const float* data = outputs[0].GetTensorData<float>();
    if (!std::all_of(data, data + info.GetElementCount(), [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("Nonfinite prompt embedding");
    return {data, data + info.GetElementCount()};
}
std::vector<float> YOLOEDetector::encode_text(const std::vector<std::string>& names) {
    open_text();
    std::vector<int64_t> ids;
    for (const auto& name : names) {
        const auto tokens = tokenizer_->tokenize(name);
        ids.insert(ids.end(), tokens.begin(), tokens.end());
    }
    const std::array<int64_t, 2> shape{static_cast<int64_t>(names.size()), 77};
    std::vector<Ort::Value> inputs;
    inputs.push_back(
        Ort::Value::CreateTensor<int64_t>(memory_info_, ids.data(), ids.size(), shape.data(), shape.size()));
    auto result = run_encoder(*text_session_, inputs, {static_cast<int64_t>(names.size()), 512}, "yoloe.text.infer");
    for (size_t i = 0; i < names.size(); ++i) {
        const auto norm =
            vision_common::normalize_embedding({result.begin() + i * 512, result.begin() + (i + 1) * 512});
        std::copy(norm.begin(), norm.end(), result.begin() + i * 512);
    }
    return result;
}
std::vector<float> YOLOEDetector::encode_visual(const vision::VisualPrompt& prompt) {
    open_visual();
    auto mask = yoloe_visual_mask(prompt, reference_side_);
    cv::Mat blob = pixels(prompt.image, reference_side_, nullptr);
    const std::array<int64_t, 4> xs{1, 3, reference_side_, reference_side_};
    const std::array<int64_t, 4> ms{1, 1, reference_side_ / 8, reference_side_ / 8};
    std::vector<Ort::Value> inputs;
    inputs.push_back(
        Ort::Value::CreateTensor<float>(memory_info_, blob.ptr<float>(), blob.total(), xs.data(), xs.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info_, mask.data(), mask.size(), ms.data(), ms.size()));
    return run_encoder(*vpe_session_, inputs, {1, 1, 512}, "yoloe.visual.infer");
}
void YOLOEDetector::ensure_prompt(
    const std::vector<std::string>& texts, const std::vector<vision::VisualPrompt>& visuals) {
    const auto names = yoloe_prompt_names(texts, visuals, defaults_);
    const bool text = visuals.empty() || !texts.empty();
    if (prompt_ready_ && text == cached_text_ && names == labels_ && same_visuals(visuals, cached_visuals_))
        return;
    std::vector<float> features;
    if (text)
        features = encode_text(names);
    if (!visuals.empty()) {
        if (!text)
            features.resize(names.size() * 512);
        for (size_t c = 0; c < names.size(); ++c) {
            std::vector<float> sum(512, 0.F);
            int examples = 0;
            for (const auto& reference : visuals) {
                if (reference.class_name != names[c])
                    continue;
                const auto feature = encode_visual(reference);
                for (size_t i = 0; i < 512; ++i)
                    sum[i] += feature[i];
                ++examples;
            }
            for (float& value : sum)
                value /= examples;
            sum = vision_common::normalize_embedding(sum);
            if (text) {
                for (size_t i = 0; i < 512; ++i) {
                    sum[i] = (sum[i] + features[c * 512 + i]) * 0.5F;
                }
                sum = vision_common::normalize_embedding(sum);
            }
            std::copy(sum.begin(), sum.end(), features.begin() + c * 512);
        }
    }
    std::vector<vision::VisualPrompt> snapshot = visuals;
    for (auto& reference : snapshot)
        reference.image = reference.image.clone();
    features.resize(kSlots * kEmbedding, 0.F);
    prompt_data_ = std::move(features);
    cached_visuals_ = std::move(snapshot);
    labels_ = names;
    cached_text_ = text;
    prompt_ready_ = true;
}
vision_core::InferResponse YOLOEDetector::RunImpl(const vision_core::InferRequest& request) {
    try {
        const auto& input = std::get<vision_core::ImageInput>(request.input);
        if (input.format != vision_core::ImagePixelFormat::kBgr8 || input.image.empty() ||
            input.image.type() != CV_8UC3) {
            throw std::invalid_argument("YOLOE detection expects a nonempty BGR8 image");
        }
        const float confidence =
            vision_core::resolve_probability_threshold(request.params.conf_threshold, confidence_, "conf_threshold");
        (void)vision_core::resolve_probability_threshold(request.params.iou_threshold, 0.5F, "iou_threshold");
        ensure_model_loaded();
        reset_runtime_profile();
        const auto begin = Clock::now();
        ensure_prompt(request.params.prompts, request.params.visual_prompts);
        auto prepared = prepare_image(
            input, image_spec(640), [](const cv::Mat& bgr, vision_operators::ImagePreprocessGeometry& geometry) {
                return pixels(bgr, 640, &geometry);
            });
        const double pre = elapsed(begin);
        const auto infer_begin = Clock::now();
        const std::array<int64_t, 4> xs{1, 3, 640, 640};
        const std::array<int64_t, 3> ps{1, 80, 512};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info_,
            const_cast<float*>(prepared.tensor().ptr<float>()), prepared.tensor().total(), xs.data(), xs.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, prompt_data_.data(), prompt_data_.size(), ps.data(), ps.size()));
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), inputs.data(), inputs.size(),
            output_node_names_.data(), 1);
        prepared.complete();
        const double infer = elapsed(infer_begin);
        add_runtime_component_timing("yoloe.detector.infer", infer);
        const auto post_begin = Clock::now();
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>{1, 84, 8400}) {
            throw std::runtime_error("YOLOE detection runtime output contract changed");
        }
        auto results = decode_yoloe_detections(outputs[0].GetTensorData<float>(), info.GetElementCount(), 80, 8400,
            static_cast<int>(labels_.size()), prepared.geometry(), confidence, request.params.max_det);
        set_runtime_preprocess_ms(pre);
        set_runtime_model_infer_ms(infer);
        set_runtime_postprocess_ms(elapsed(post_begin));
        set_runtime_total_ms(elapsed(begin));
        return vision_core::make_infer_response(std::move(results));
    } catch (const std::invalid_argument& error) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = error.what();
        response.error_code = vision_core::InferErrorCode::kInvalidArgument;
        return response;
    }
}
void YOLOEDetector::release() {
    text_session_.reset();
    vpe_session_.reset();
    tokenizer_.reset();
    cached_visuals_.clear();
    prompt_data_.clear();
    labels_.clear();
    prompt_ready_ = false;
    BaseModel::release();
}
std::vector<vision_core::InferIntent> YOLOEDetector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}
std::vector<vision_core::ModelCapability> YOLOEDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}
std::vector<std::string> YOLOEDetector::get_dynamic_class_names() const { return labels_; }
static vision_core::ModelRegistrar<YOLOEDetector> registrar("YOLOEDetector");
} // namespace vision_deploy
