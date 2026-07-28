/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ppocr_detector.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/opencv.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>  // OpenCV 5: contour/hull/minAreaRect/getPerspectiveTransform
#endif

#include "spacemit_ort_env.h"  // NOLINT(build/include_order)

#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> PPOCRDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for PPOCRDetector");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for PPOCRDetector");
    }

    std::string rec_model_path = vision_core::yaml_utils::getString(default_params, "rec_model_path");
    if (rec_model_path.empty()) {
        throw std::runtime_error("rec_model_path not found in default_params for PPOCRDetector");
    }
    std::string dict_path = vision_core::yaml_utils::getString(default_params, "dict_path");
    if (dict_path.empty()) {
        throw std::runtime_error("dict_path not found in default_params for PPOCRDetector");
    }

    int det_limit_side_len = vision_core::yaml_utils::getInt(default_params, "det_limit_side_len", 960);
    int det_input_h = vision_core::yaml_utils::getInt(default_params, "det_input_h", 0);
    int det_input_w = vision_core::yaml_utils::getInt(default_params, "det_input_w", 0);
    float det_db_thresh = vision_core::yaml_utils::getFloat(default_params, "det_db_thresh", 0.3f);
    float det_db_box_thresh = vision_core::yaml_utils::getFloat(default_params, "det_db_box_thresh", 0.6f);
    float det_db_unclip_ratio = vision_core::yaml_utils::getFloat(default_params, "det_db_unclip_ratio", 2.0f);
    float det_box_nms_thresh = vision_core::yaml_utils::getFloat(default_params, "det_box_nms_thresh", 0.5f);
    int rec_img_h = vision_core::yaml_utils::getInt(default_params, "rec_img_h", 48);
    int rec_img_w_max = vision_core::yaml_utils::getInt(default_params, "rec_img_w_max", 320);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<PPOCRDetector>(
        model_path, rec_model_path, dict_path, det_limit_side_len, det_input_h, det_input_w,
        det_db_thresh, det_db_box_thresh, det_db_unclip_ratio, det_box_nms_thresh,
        rec_img_h, rec_img_w_max, num_threads, lazy_load, provider);
}

PPOCRDetector::PPOCRDetector(
    const std::string& model_path,
    const std::string& rec_model_path,
    const std::string& dict_path,
    int det_limit_side_len,
    int det_input_h,
    int det_input_w,
    float det_db_thresh,
    float det_db_box_thresh,
    float det_db_unclip_ratio,
    float det_box_nms_thresh,
    int rec_img_h,
    int rec_img_w_max,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        rec_model_path_(rec_model_path),
        dict_path_(dict_path),
        det_limit_side_len_(det_limit_side_len),
        det_input_h_(det_input_h),
        det_input_w_(det_input_w),
        det_db_thresh_(det_db_thresh),
        det_db_box_thresh_(det_db_box_thresh),
        det_db_unclip_ratio_(det_db_unclip_ratio),
        det_box_nms_thresh_(det_box_nms_thresh),
        rec_img_h_(rec_img_h),
        rec_img_w_max_(rec_img_w_max),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void PPOCRDetector::load_dict(const std::string& dict_path) {
    std::ifstream f(dict_path);
    if (!f.is_open()) {
        throw std::runtime_error("PPOCRDetector: cannot open dict file: " + dict_path);
    }
    // Read character lines. CTC blank is index 0:
    // - PP-OCRv5 keys already start with "blank"
    // - PP-OCRv6 tiny dict starts with the first printable char; prepend blank
    dict_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // tolerate CRLF
        }
        // An empty dict line represents the space character.
        dict_.push_back(line.empty() ? " " : line);
    }
    f.close();
    if (dict_.size() <= 1) {
        throw std::runtime_error("PPOCRDetector: dict file is empty: " + dict_path);
    }
    if (dict_.front() != "blank") {
        dict_.insert(dict_.begin(), "blank");
    }
    // use_space_char: PaddleOCR appends a trailing space class after dict entries.
    dict_.push_back(" ");
}

void PPOCRDetector::validate_dict_size() {
    if (!rec_session_ || dict_.empty()) {
        return;
    }
    const auto shape =
        rec_session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[2] <= 0) {
        return;  // dynamic class count; skip static check
    }
    const size_t expected = static_cast<size_t>(shape[2]);
    if (dict_.size() != expected) {
        throw std::runtime_error(
            "PPOCRDetector: dict size mismatch (dict has " + std::to_string(dict_.size()) +
            " classes including blank/space, rec model expects " + std::to_string(expected) +
            "). Check dict_path matches the rec model (blank + chars + space).");
    }
}

void PPOCRDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    // Detection session (BaseModel::session_). Input is dynamic HxW, so
    // input_shape_ from init_session is unreliable; detect_text builds the shape
    // per image and calls session_->Run directly.
    // Det and rec both honor yaml default_params.providers (via provider_).
    init_session(num_threads_, provider_);

    Ort::SessionOptions rec_opts;
    rec_opts.SetIntraOpNumThreads(num_threads_);
    rec_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider_ == "SpaceMITExecutionProvider") {
        Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(rec_opts);
        if (!status.IsOK()) {
            throw std::runtime_error(
                std::string("SpaceMIT EP init failed (rec): ") + status.GetErrorMessage());
        }
    }
    std::cout << provider_ << " (rec): " << rec_model_path_ << std::endl;
    // rec_model_path_ / dict_path_ arrive already resolved: the factory expands
    // any default_params key ending in "_path" via resolveResourcePath before create().
    rec_session_ = std::make_unique<Ort::Session>(
        vision_core::shared_ort_env(), rec_model_path_.c_str(), rec_opts);

    const size_t rec_num_in = rec_session_->GetInputCount();
    rec_input_names_.resize(rec_num_in);
    rec_input_names_str_.resize(rec_num_in);
    for (size_t i = 0; i < rec_num_in; ++i) {
        auto n = rec_session_->GetInputNameAllocated(i, allocator_);
        rec_input_names_str_[i] = n.get();
        rec_input_names_[i] = rec_input_names_str_[i].c_str();
    }
    const size_t rec_num_out = rec_session_->GetOutputCount();
    rec_output_names_.resize(rec_num_out);
    rec_output_names_str_.resize(rec_num_out);
    for (size_t i = 0; i < rec_num_out; ++i) {
        auto n = rec_session_->GetOutputNameAllocated(i, allocator_);
        rec_output_names_str_[i] = n.get();
        rec_output_names_[i] = rec_output_names_str_[i].c_str();
    }

    load_dict(dict_path_);
    validate_dict_size();
    model_loaded_ = true;
}

// ---------------------------------------------------------------------------
// Detection (DBNet)
// ---------------------------------------------------------------------------

std::vector<float> PPOCRDetector::det_preprocess(const cv::Mat& image, int* net_h, int* net_w) {
    const int ori_h = image.rows;
    const int ori_w = image.cols;

    int rh = 0;
    int rw = 0;
    int canvas_h = 0;
    int canvas_w = 0;
    if (det_input_h_ > 0 && det_input_w_ > 0) {
        // Fixed-shape det (NPU): keep aspect ratio (bilinear), then pad to HxW.
        // Content is placed at top-left; pad is zeros on the right/bottom.
        const float scale = std::min(
            static_cast<float>(det_input_w_) / static_cast<float>(ori_w),
            static_cast<float>(det_input_h_) / static_cast<float>(ori_h));
        rw = std::max(1, static_cast<int>(std::round(ori_w * scale)));
        rh = std::max(1, static_cast<int>(std::round(ori_h * scale)));
        rw = std::min(rw, det_input_w_);
        rh = std::min(rh, det_input_h_);
        canvas_h = det_input_h_;
        canvas_w = det_input_w_;
    } else {
        // Dynamic det: keep aspect ratio so the longer side <= det_limit_side_len,
        // then align each side up to a multiple of 32 (DBNet stride requirement).
        float ratio = static_cast<float>(std::max(ori_h, ori_w)) /
            static_cast<float>(det_limit_side_len_);
        rh = ori_h;
        rw = ori_w;
        if (ratio > 1.0f) {
            rh = static_cast<int>(ori_h / ratio);
            rw = static_cast<int>(ori_w / ratio);
        }
        rh = (rh + 31) / 32 * 32;
        rw = (rw + 31) / 32 * 32;
        canvas_h = rh;
        canvas_w = rw;
    }
    det_resize_h_ = rh;
    det_resize_w_ = rw;
    *net_h = canvas_h;
    *net_w = canvas_w;

    // Resize+pad on uint8 first (cheaper than float), then blobFromImage for
    // BGR->RGB /255 / HWC->CHW, then ImageNet mean/std on CHW.
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded;
    if (canvas_h == rh && canvas_w == rw) {
        padded = resized;
    } else {
        padded = cv::Mat::zeros(canvas_h, canvas_w, CV_8UC3);
        resized.copyTo(padded(cv::Rect(0, 0, rw, rh)));
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        padded, 1.0 / 255.0, cv::Size(), cv::Scalar(),
        /*swapRB=*/true, /*crop=*/false, CV_32F);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std_val[3] = {0.229f, 0.224f, 0.225f};
    const int plane = canvas_h * canvas_w;
    float* blob_data = blob.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        float* ch = blob_data + static_cast<size_t>(c) * plane;
        const float inv_std = 1.0f / std_val[c];
        for (int i = 0; i < plane; ++i) {
            ch[i] = (ch[i] - mean[c]) * inv_std;
        }
    }

    return std::vector<float>(blob_data, blob_data + static_cast<size_t>(3) * plane);
}

std::vector<cv::Point> PPOCRDetector::unclip(const std::vector<cv::Point>& poly) {
    // Expand polygon outward by distance = |area| * unclip_ratio / perimeter
    // (PaddleOCR DBPostProcess / pyclipper offset).
    //
    // Image coordinates have y pointing down, so the mathematical left-hand
    // edge normal points *inward* for the usual CCW (positive contourArea)
    // winding from OpenCV boxPoints. Choose the normal that points away from
    // the polygon centroid so expansion is always outward.
    const double area = std::abs(cv::contourArea(poly));
    const double length = cv::arcLength(poly, true);
    if (length < 1e-6 || area < 1e-6) {
        return poly;
    }
    const double dist = area * det_db_unclip_ratio_ / length;

    cv::Point2f center(0.f, 0.f);
    for (const cv::Point& p : poly) {
        center.x += static_cast<float>(p.x);
        center.y += static_cast<float>(p.y);
    }
    center.x /= static_cast<float>(poly.size());
    center.y /= static_cast<float>(poly.size());

    const size_t n = poly.size();
    std::vector<cv::Point2f> hull_in;
    hull_in.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        cv::Point2f p1(static_cast<float>(poly[i].x), static_cast<float>(poly[i].y));
        cv::Point2f p2(static_cast<float>(poly[(i + 1) % n].x),
            static_cast<float>(poly[(i + 1) % n].y));
        cv::Point2f edge = p2 - p1;
        const float len = std::sqrt(edge.x * edge.x + edge.y * edge.y);
        if (len < 1e-6f) {
            continue;
        }
        cv::Point2f normal(-edge.y / len, edge.x / len);
        const cv::Point2f mid = (p1 + p2) * 0.5f;
        if (normal.dot(mid - center) < 0.f) {
            normal = -normal;
        }
        hull_in.push_back(p1 + normal * static_cast<float>(dist));
        hull_in.push_back(p2 + normal * static_cast<float>(dist));
    }
    std::vector<cv::Point> hull;
    if (!hull_in.empty()) {
        std::vector<cv::Point2f> hull_f;
        cv::convexHull(hull_in, hull_f);
        hull.reserve(hull_f.size());
        for (const cv::Point2f& p : hull_f) {
            hull.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        }
    }
    return hull;
}

float PPOCRDetector::box_score(const cv::Mat& prob_map, const std::vector<cv::Point>& box) {
    const int h = prob_map.rows;
    const int w = prob_map.cols;
    std::vector<cv::Point> pts;
    pts.reserve(box.size());
    for (const auto& p : box) {
        pts.push_back({std::max(0, std::min(w - 1, p.x)), std::max(0, std::min(h - 1, p.y))});
    }
    cv::Rect bbox = cv::boundingRect(pts);
    bbox.x = std::max(0, bbox.x);
    bbox.y = std::max(0, bbox.y);
    bbox.width = std::min(w - bbox.x, bbox.width);
    bbox.height = std::min(h - bbox.y, bbox.height);
    if (bbox.width <= 0 || bbox.height <= 0) {
        return 0.0f;
    }
    cv::Mat mask = cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1);
    std::vector<cv::Point> shifted;
    shifted.reserve(pts.size());
    for (const auto& p : pts) {
        shifted.push_back({p.x - bbox.x, p.y - bbox.y});
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted}, cv::Scalar(1));
    return static_cast<float>(cv::mean(prob_map(bbox), mask)[0]);
}

// Order 4 corners to TL, TR, BR, BL (matches PaddleOCR get_mini_boxes).
void PPOCRDetector::sort_box_points(std::vector<cv::Point>& pts) {
    std::sort(pts.begin(), pts.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });
    const int tl = pts[0].y <= pts[1].y ? 0 : 1;
    const int bl = 1 - tl;
    const int tr = pts[2].y <= pts[3].y ? 2 : 3;
    const int br = (tr == 2) ? 3 : 2;
    std::vector<cv::Point> sorted = {pts[tl], pts[tr], pts[br], pts[bl]};
    pts = sorted;
}

std::vector<PPOCRDetector::TextBox> PPOCRDetector::db_postprocess(
    const cv::Mat& prob_map, int ori_h, int ori_w, int net_h, int net_w) {
    // Binarize, then dilate 3x3 to connect nearby strokes (PaddleOCR default).
    cv::Mat bin_map;
    cv::threshold(prob_map, bin_map, det_db_thresh_, 255.0, cv::THRESH_BINARY);
    bin_map.convertTo(bin_map, CV_8UC1);
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::dilate(bin_map, dilated, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilated, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Map boxes from network space back to the original image. For letterbox
    // (fixed det_input_*), scale uses the content size before pad, not net HxW.
    const int map_h = (det_resize_h_ > 0) ? det_resize_h_ : net_h;
    const int map_w = (det_resize_w_ > 0) ? det_resize_w_ : net_w;
    const float scale_x = static_cast<float>(ori_w) / static_cast<float>(map_w);
    const float scale_y = static_cast<float>(ori_h) / static_cast<float>(map_h);

    std::vector<TextBox> boxes;
    for (const auto& contour : contours) {
        if (contour.size() < 4) {
            continue;
        }
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (std::min(rect.size.width, rect.size.height) < 3.0f) {
            continue;
        }
        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<cv::Point> box_pts(4);
        for (int i = 0; i < 4; ++i) {
            box_pts[i] = {static_cast<int>(pts[i].x), static_cast<int>(pts[i].y)};
        }

        const float score = box_score(prob_map, box_pts);
        if (score < det_db_box_thresh_) {
            continue;
        }

        std::vector<cv::Point> expanded = unclip(box_pts);
        if (expanded.size() < 4) {
            continue;
        }
        cv::RotatedRect exp_rect = cv::minAreaRect(expanded);
        cv::Point2f exp_pts[4];
        exp_rect.points(exp_pts);

        std::vector<cv::Point> final_pts(4);
        for (int i = 0; i < 4; ++i) {
            final_pts[i] = {
                std::max(0, std::min(ori_w - 1, static_cast<int>(std::round(exp_pts[i].x * scale_x)))),
                std::max(0, std::min(ori_h - 1, static_cast<int>(std::round(exp_pts[i].y * scale_y))))};
        }
        sort_box_points(final_pts);

        TextBox tb;
        tb.points = std::move(final_pts);
        tb.score = score;
        boxes.push_back(std::move(tb));
    }
    return boxes;
}

std::vector<PPOCRDetector::TextBox> PPOCRDetector::nms_boxes(
    std::vector<TextBox> boxes, float nms_thresh) {
    if (boxes.size() <= 1 || nms_thresh <= 0.0f) {
        return boxes;
    }

    std::sort(boxes.begin(), boxes.end(),
        [](const TextBox& a, const TextBox& b) { return a.score > b.score; });

    auto aabb = [](const std::vector<cv::Point>& pts) {
        cv::Rect r = cv::boundingRect(pts);
        return r;
    };
    auto iou = [](const cv::Rect& a, const cv::Rect& b) {
        const int x1 = std::max(a.x, b.x);
        const int y1 = std::max(a.y, b.y);
        const int x2 = std::min(a.x + a.width, b.x + b.width);
        const int y2 = std::min(a.y + a.height, b.y + b.height);
        const int inter_w = std::max(0, x2 - x1);
        const int inter_h = std::max(0, y2 - y1);
        const int inter = inter_w * inter_h;
        const int uni = a.width * a.height + b.width * b.height - inter;
        return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
    };

    std::vector<TextBox> kept;
    kept.reserve(boxes.size());
    std::vector<char> suppressed(boxes.size(), 0);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(boxes[i]);
        const cv::Rect ri = aabb(boxes[i].points);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (iou(ri, aabb(boxes[j].points)) > nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------
// Recognition (CRNN + CTC)
// ---------------------------------------------------------------------------

cv::Mat PPOCRDetector::crop_text_box(const cv::Mat& image, const std::vector<cv::Point>& box) {
    // box ordered TL, TR, BR, BL. Perspective-warp to an axis-aligned crop.
    cv::Point2f src[4];
    for (int i = 0; i < 4; ++i) {
        src[i] = cv::Point2f(static_cast<float>(box[i].x), static_cast<float>(box[i].y));
    }
    const float w1 = std::hypot(src[1].x - src[0].x, src[1].y - src[0].y);
    const float w2 = std::hypot(src[2].x - src[3].x, src[2].y - src[3].y);
    const float h1 = std::hypot(src[3].x - src[0].x, src[3].y - src[0].y);
    const float h2 = std::hypot(src[2].x - src[1].x, src[2].y - src[1].y);
    const int crop_w = std::max(1, static_cast<int>(std::max(w1, w2)));
    const int crop_h = std::max(1, static_cast<int>(std::max(h1, h2)));

    cv::Point2f dst[4] = {
        {0.f, 0.f},
        {static_cast<float>(crop_w - 1), 0.f},
        {static_cast<float>(crop_w - 1), static_cast<float>(crop_h - 1)},
        {0.f, static_cast<float>(crop_h - 1)}};
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat crop;
    cv::warpPerspective(image, crop, M, {crop_w, crop_h});

    // Vertical text: rotate to horizontal (PaddleOCR uses counter-clockwise).
    if (crop_h > crop_w * 1.5f) {
        cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    return crop;
}

cv::Mat PPOCRDetector::rec_make_canvas(const cv::Mat& crop) const {
    // Keep aspect ratio, pad to [rec_img_h_ x rec_img_w_max_] with gray 127
    // (PaddleOCR RecResizeImg).
    cv::Mat canvas(rec_img_h_, rec_img_w_max_, CV_8UC3, cv::Scalar(127, 127, 127));
    if (crop.empty() || crop.rows <= 0 || crop.cols <= 0) {
        return canvas;
    }
    int target_w = std::max(1, static_cast<int>(
        static_cast<float>(crop.cols) / static_cast<float>(crop.rows) * rec_img_h_));
    target_w = std::min(target_w, rec_img_w_max_);
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(target_w, rec_img_h_), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(0, 0, target_w, rec_img_h_)));
    return canvas;
}

std::string PPOCRDetector::ctc_decode(
    const float* logits, int seq_len, int num_classes, float* out_score) const {
    std::string text;
    float score_sum = 0.0f;
    int score_cnt = 0;
    int last_idx = -1;
    for (int t = 0; t < seq_len; ++t) {
        const float* step = logits + static_cast<size_t>(t) * num_classes;
        const int best = static_cast<int>(std::max_element(step, step + num_classes) - step);
        if (best != 0 && best != last_idx) {
            if (best < static_cast<int>(dict_.size())) {
                text += dict_[static_cast<size_t>(best)];
                score_sum += step[best];
                ++score_cnt;
            }
        }
        last_idx = best;
    }
    if (out_score) {
        *out_score = score_cnt > 0 ? score_sum / static_cast<float>(score_cnt) : 0.0f;
    }
    return text;
}

std::string PPOCRDetector::rec_run(
    const cv::Mat& crop,
    float* out_score,
    double* model_infer_ms,
    uint64_t* model_infer_calls) {
    *out_score = 0.0f;
    *model_infer_ms = 0.0;
    *model_infer_calls = 0;
    if (crop.empty()) {
        return "";
    }

    cv::Mat canvas = rec_make_canvas(crop);
    // (x - 127.5) / 127.5; mean order is (R,G,B) with swapRB.
    cv::Mat blob = cv::dnn::blobFromImage(
        canvas, 1.0 / 127.5, cv::Size(), cv::Scalar(127.5, 127.5, 127.5),
        /*swapRB=*/true, /*crop=*/false, CV_32F);

    const int plane = rec_img_h_ * rec_img_w_max_;
    const std::vector<int64_t> shape = {1, 3, rec_img_h_, rec_img_w_max_};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info_, blob.ptr<float>(), static_cast<size_t>(3) * plane,
        shape.data(), shape.size());
    const auto t_rec_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outs = rec_session_->Run(
        Ort::RunOptions{nullptr}, rec_input_names_.data(), &input, 1,
        rec_output_names_.data(), rec_output_names_.size());
    const auto t_rec_infer1 = std::chrono::steady_clock::now();
    *model_infer_calls = 1;
    *model_infer_ms =
        std::chrono::duration<double, std::milli>(
            t_rec_infer1 - t_rec_infer0).count();

    // Output [1, T, num_classes]; CTC greedy decode (blank == index 0).
    const float* logits = outs[0].GetTensorData<float>();
    std::vector<int64_t> dims = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 3) {
        return "";
    }
    return ctc_decode(logits, static_cast<int>(dims[1]), static_cast<int>(dims[2]), out_score);
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

vision_common::TextResultList PPOCRDetector::detect_text(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("PPOCRDetector: input image is empty");
    }
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // --- detection ---
    const auto t_pre0 = std::chrono::steady_clock::now();
    int net_h = 0, net_w = 0;
    std::vector<float> det_input = det_preprocess(image, &net_h, &net_w);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_inf0 = std::chrono::steady_clock::now();
    const std::vector<int64_t> det_shape = {1, 3, net_h, net_w};
    Ort::Value det_in = Ort::Value::CreateTensor<float>(
        memory_info_, det_input.data(), det_input.size(), det_shape.data(), det_shape.size());
    std::vector<Ort::Value> det_out = session_->Run(
        Ort::RunOptions{nullptr}, input_node_names_.data(), &det_in, 1,
        output_node_names_.data(), output_node_names_.size());
    const auto t_inf1 = std::chrono::steady_clock::now();
    const double detector_infer_ms =
        std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();
    add_runtime_component_timing("detector.infer", detector_infer_ms);

    const auto t_post0 = std::chrono::steady_clock::now();
    float* prob = det_out[0].GetTensorMutableData<float>();
    std::vector<int64_t> pdims = det_out[0].GetTensorTypeAndShapeInfo().GetShape();
    // prob shape [1, 1, H, W]
    const int prob_h = static_cast<int>(pdims[pdims.size() - 2]);
    const int prob_w = static_cast<int>(pdims[pdims.size() - 1]);
    cv::Mat prob_map(prob_h, prob_w, CV_32FC1, prob);
    std::vector<TextBox> boxes = db_postprocess(prob_map, image.rows, image.cols, net_h, net_w);
    boxes = nms_boxes(std::move(boxes), det_box_nms_thresh_);

    // --- recognition per box ---
    vision_common::TextResultList results;
    results.reserve(boxes.size());
    double recognizer_infer_total_ms = 0.0;
    for (const TextBox& box : boxes) {
        cv::Mat crop = crop_text_box(image, box.points);
        float rec_score = 0.0f;
        double recognizer_infer_ms = 0.0;
        uint64_t recognizer_infer_calls = 0;
        std::string text = rec_run(
            crop,
            &rec_score,
            &recognizer_infer_ms,
            &recognizer_infer_calls);
        if (recognizer_infer_calls > 0) {
            add_runtime_component_timing(
                "recognizer.infer",
                recognizer_infer_ms,
                recognizer_infer_calls);
            recognizer_infer_total_ms += recognizer_infer_ms;
        }
        if (text.empty()) {
            continue;
        }
        vision_common::TextResult tr;
        tr.polygon.reserve(box.points.size());
        for (const cv::Point& p : box.points) {
            vision::KeyPoint kp;
            kp.x = static_cast<float>(p.x);
            kp.y = static_cast<float>(p.y);
            kp.visibility = 1.0f;
            tr.polygon.push_back(kp);
        }
        tr.text = std::move(text);
        tr.score = rec_score;
        tr.label = -1;
        results.push_back(std::move(tr));
    }
    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(
        detector_infer_ms + recognizer_infer_total_ms);
    const double postprocess_with_rec_infer_ms =
        std::chrono::duration<double, std::milli>(t1 - t_post0).count();
    set_runtime_postprocess_ms(std::max(
        0.0, postprocess_with_rec_infer_ms - recognizer_infer_total_ms));
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::InferIntent> PPOCRDetector::supported_intents() const {
    return {vision_core::InferIntent::kOcr};
}

std::vector<vision_core::ModelCapability> PPOCRDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_core::InferResponse PPOCRDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kOcr);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "PPOCRDetector expects ImageInput";
        return response;
    }
    vision_common::TextResultList task_results = detect_text(image_input->image);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

static vision_core::ModelRegistrar<PPOCRDetector> registrar("PPOCRDetector");

}  // namespace vision_deploy
