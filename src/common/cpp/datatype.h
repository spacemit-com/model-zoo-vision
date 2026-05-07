/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DATATYPE_H
#define DATATYPE_H

#include <memory>
#include <vector>
#include <variant>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

// Forward declaration to avoid including OpenCV headers
namespace cv {
    class Mat;
}

/**
 * @file datatype.h
 * @brief Type-safe result structures using std::variant
 *
 * This file defines strongly-typed result structures for different vision tasks.
 * Each task type has its own dedicated structure, eliminating the "god struct" problem.
 */

namespace vision_common {

// ============================================================================
// Basic Types
// ============================================================================

/**
 * @brief Bounding box representation
 */
struct BoundingBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float center_x() const { return (x1 + x2) / 2.0f; }
    float center_y() const { return (y1 + y2) / 2.0f; }

    bool is_valid() const {
        // Allow negative coordinates (some models may output them)
        // Only check that bbox has positive area
        return x2 > x1 && y2 > y1;
    }

    // IoU calculation (pixel coordinates with +1, consistent with Python)
    float iou(const BoundingBox& other) const {
        float inter_x1 = std::max(x1, other.x1);
        float inter_y1 = std::max(y1, other.y1);
        float inter_x2 = std::min(x2, other.x2);
        float inter_y2 = std::min(y2, other.y2);

        float inter_w = std::max(0.0f, inter_x2 - inter_x1 + 1.0f);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1 + 1.0f);
        float inter_area = inter_w * inter_h;

        float area1 = (x2 - x1 + 1.0f) * (y2 - y1 + 1.0f);
        float area2 = (other.x2 - other.x1 + 1.0f) * (other.y2 - other.y1 + 1.0f);
        float union_area = area1 + area2 - inter_area;

        return union_area > 0 ? inter_area / union_area : 0.0f;
    }
};

/**
 * @brief Keypoint for pose estimation
 */
struct KeyPoint {
    float x = 0.0f;
    float y = 0.0f;
    float visibility = 0.0f;  // 0.0-1.0, confidence score

    bool is_visible() const { return visibility > 0.5f; }
    bool is_valid() const { return x >= 0 && y >= 0; }
};

// ============================================================================
// Task-Specific Result Types
// ============================================================================

/**
 * @brief Object detection result
 * Used by: YOLOv5, YOLOv8, YOLOv11, YOLOv5-face, YOLOv5-gesture
 */
struct DetectionResult {
    BoundingBox bbox;
    float score = 0.0f;
    int label = -1;

    bool is_valid() const {
        // Relaxed: allow label=-1 (unlabeled detections)
        return bbox.is_valid() && score > 0;
    }
};

/**
 * @brief Image classification result
 * Used by: ResNet, Emotion
 */
struct ClassificationResult {
    int label = -1;
    float score = 0.0f;
    std::vector<float> class_scores;  // All class probabilities

    bool is_valid() const {
        return label >= 0 && score > 0;
    }

    // Get top-k predictions
    std::vector<std::pair<int, float>> top_k(int k) const {
        std::vector<std::pair<int, float>> indexed_scores;
        indexed_scores.reserve(class_scores.size());
        for (size_t i = 0; i < class_scores.size(); ++i) {
            indexed_scores.emplace_back(static_cast<int>(i), class_scores[i]);
        }

        std::partial_sort(
            indexed_scores.begin(),
            indexed_scores.begin() + std::min(k, static_cast<int>(indexed_scores.size())),
            indexed_scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        indexed_scores.resize(std::min(k, static_cast<int>(indexed_scores.size())));
        return indexed_scores;
    }
};

/**
 * @brief Pose estimation result
 * Used by: YOLOv8-pose
 */
struct PoseResult {
    BoundingBox bbox;
    float score = 0.0f;
    int label = -1;  // Person class or pose type
    std::vector<KeyPoint> keypoints;

    bool is_valid() const {
        // Relaxed: allow empty keypoints (detection may fail)
        return bbox.is_valid() && score > 0;
    }

    bool has_keypoint(size_t idx) const {
        return idx < keypoints.size() && keypoints[idx].is_visible();
    }

    size_t visible_keypoint_count() const {
        return std::count_if(
            keypoints.begin(), keypoints.end(),
            [](const KeyPoint& kp) { return kp.is_visible(); });
    }
};

/**
 * @brief Instance segmentation result
 * Used by: YOLOv8-seg, PP-LiteSeg
 */
struct SegmentationResult {
    BoundingBox bbox;
    float score = 0.0f;
    int label = -1;
    std::shared_ptr<cv::Mat> mask;  // Binary mask

    bool is_valid() const {
        // Relaxed: allow label=-1, mask can be null for bbox-only results
        return bbox.is_valid() && score > 0;
    }

    bool has_mask() const { return mask != nullptr; }
};

/**
 * @brief Face/object embedding result
 * Used by: ArcFace
 */
struct EmbeddingResult {
    std::vector<float> embedding;
    float score = 1.0f;  // Optional: embedding quality score

    bool is_valid() const {
        return !embedding.empty();
    }

    size_t dimension() const { return embedding.size(); }

    // Cosine similarity with another embedding
    float similarity(const EmbeddingResult& other) const {
        if (embedding.size() != other.embedding.size() || embedding.empty()) {
            return 0.0f;
        }

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < embedding.size(); ++i) {
            dot += embedding[i] * other.embedding[i];
            norm_a += embedding[i] * embedding[i];
            norm_b += other.embedding[i] * other.embedding[i];
        }

        float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return denom > 0 ? dot / denom : 0.0f;
    }

    // L2 distance
    float distance(const EmbeddingResult& other) const {
        if (embedding.size() != other.embedding.size() || embedding.empty()) {
            return std::numeric_limits<float>::max();
        }

        float sum = 0.0f;
        for (size_t i = 0; i < embedding.size(); ++i) {
            float diff = embedding[i] - other.embedding[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

/**
 * @brief Object tracking result
 * Used by: ByteTrack, OC-SORT
 */
struct TrackingResult {
    BoundingBox bbox;
    float score = 0.0f;
    int label = -1;
    int track_id = -1;

    // Optional: tracking state
    enum class State : uint8_t {
        Tentative = 0,  // New track, not confirmed yet
        Confirmed = 1,  // Stable track
        Lost = 2        // Track lost
    };
    State state = State::Confirmed;

    bool is_valid() const {
        // Relaxed: allow track_id=-1 (untracked objects)
        return bbox.is_valid() && score > 0;
    }

    bool is_confirmed() const { return state == State::Confirmed; }
};

/**
 * @brief Action recognition result (sequence-based)
 * Used by: STGCN
 */
struct ActionResult {
    int label = -1;  // Renamed from action_class for consistency
    float score = 0.0f;
    std::vector<float> class_scores;  // All action class probabilities

    bool is_valid() const {
        return label >= 0 && score > 0;
    }
};

// ============================================================================
// Variant Definition
// ============================================================================

/**
 * @brief Unified result type using std::variant
 *
 * This variant can hold any of the task-specific result types.
 * Use std::visit or std::get_if to access the contained value.
 */
using ModelResult = std::variant<
    DetectionResult,
    ClassificationResult,
    PoseResult,
    SegmentationResult,
    EmbeddingResult,
    TrackingResult,
    ActionResult
>;

// ============================================================================
// Type Aliases (for interface compatibility)
// ============================================================================

using DetectionResultList = std::vector<DetectionResult>;
using ClassificationResultList = std::vector<ClassificationResult>;
using PoseResultList = std::vector<PoseResult>;
using SegmentationResultList = std::vector<SegmentationResult>;
using TrackingResultList = std::vector<TrackingResult>;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Type checking helpers
 */
inline bool is_detection(const ModelResult& r) {
    return std::holds_alternative<DetectionResult>(r);
}

inline bool is_classification(const ModelResult& r) {
    return std::holds_alternative<ClassificationResult>(r);
}

inline bool is_pose(const ModelResult& r) {
    return std::holds_alternative<PoseResult>(r);
}

inline bool is_segmentation(const ModelResult& r) {
    return std::holds_alternative<SegmentationResult>(r);
}

inline bool is_embedding(const ModelResult& r) {
    return std::holds_alternative<EmbeddingResult>(r);
}

inline bool is_tracking(const ModelResult& r) {
    return std::holds_alternative<TrackingResult>(r);
}

inline bool is_action(const ModelResult& r) {
    return std::holds_alternative<ActionResult>(r);
}

/**
 * @brief Type-safe accessors (returns nullptr if wrong type)
 */
inline const DetectionResult* as_detection(const ModelResult& r) {
    return std::get_if<DetectionResult>(&r);
}

inline const ClassificationResult* as_classification(const ModelResult& r) {
    return std::get_if<ClassificationResult>(&r);
}

inline const PoseResult* as_pose(const ModelResult& r) {
    return std::get_if<PoseResult>(&r);
}

inline const SegmentationResult* as_segmentation(const ModelResult& r) {
    return std::get_if<SegmentationResult>(&r);
}

inline const EmbeddingResult* as_embedding(const ModelResult& r) {
    return std::get_if<EmbeddingResult>(&r);
}

inline const TrackingResult* as_tracking(const ModelResult& r) {
    return std::get_if<TrackingResult>(&r);
}

inline const ActionResult* as_action(const ModelResult& r) {
    return std::get_if<ActionResult>(&r);
}

/**
 * @brief Get common properties (works for most result types)
 */
inline float get_score(const ModelResult& r) {
    return std::visit([](const auto& result) -> float {
        return result.score;
    }, r);
}

inline BoundingBox get_bbox(const ModelResult& r) {
    return std::visit([](const auto& result) -> BoundingBox {
        using T = std::decay_t<decltype(result)>;
        if constexpr (
            std::is_same_v<T, DetectionResult> ||
            std::is_same_v<T, PoseResult> ||
            std::is_same_v<T, SegmentationResult> ||
            std::is_same_v<T, TrackingResult>) {
            return result.bbox;
        } else {
            return BoundingBox{};
        }
    }, r);
}

inline int get_label(const ModelResult& r) {
    return std::visit([](const auto& result) -> int {
        using T = std::decay_t<decltype(result)>;
        if constexpr (
            std::is_same_v<T, DetectionResult> ||
            std::is_same_v<T, ClassificationResult> ||
            std::is_same_v<T, PoseResult> ||
            std::is_same_v<T, SegmentationResult> ||
            std::is_same_v<T, TrackingResult> ||
            std::is_same_v<T, ActionResult>) {
            return result.label;
        } else {
            return -1;
        }
    }, r);
}

/**
 * @brief Generic visitor helper
 */
template<typename Visitor>
auto visit_result(const ModelResult& r, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), r);
}

}  // namespace vision_common

#endif  // DATATYPE_H
