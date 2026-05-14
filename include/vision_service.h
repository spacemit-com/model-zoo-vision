/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Public API / stable ABI: this header and the vision library are the customer-facing
 * contract. Internal refactors (see docs/design_roadmap.md) do not change this API.
 */

#ifndef VISION_SERVICE_H
#define VISION_SERVICE_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct VisionServiceKeypoint {
    float x;
    float y;
    float visibility;
};

struct VisionServiceResult {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    int track_id;
    std::vector<VisionServiceKeypoint> keypoints;  // empty if not a pose result
    cv::Mat mask;                                    // empty if not a segmentation result
};

struct VisionServiceTiming {
    // Generic image pipeline stages (most models)
    double preprocess_ms = 0.0;
    double model_infer_ms = 0.0;
    double postprocess_ms = 0.0;

    // Tracking pipeline stages (tracking models)
    double detect_ms = 0.0;
    double track_ms = 0.0;

    // Aggregated totals
    double infer_ms = 0.0;
    double draw_ms = 0.0;

    // Other task types
    double embedding_ms = 0.0;
    double sequence_ms = 0.0;
};

struct VisionServiceTimingOptions {
    bool enabled = false;
    bool print_to_stdout = false;
};

enum VisionServiceStatus {
    VISION_SERVICE_OK = 0,
    VISION_SERVICE_INVALID_ARGUMENT = 1,
    VISION_SERVICE_CREATE_FAILED = 2,
    VISION_SERVICE_INFER_FAILED = 3,
    VISION_SERVICE_IO_FAILED = 4
};

/**
 * @brief Vision inference service
 *
 * @thread_safety IMPORTANT: VisionService instances are NOT thread-safe.
 *
 * Thread Safety Rules:
 * - Create() is thread-safe (can be called from multiple threads)
 * - All other methods are NOT thread-safe for the same instance
 * - Do NOT call InferImage/InferEmbedding/Draw concurrently on the same instance
 *
 * Correct Multi-threading Usage:
 * - Option 1: Create separate VisionService instances for each thread (RECOMMENDED)
 * - Option 2: Use a thread pool with one VisionService per worker thread
 * - Option 3: Protect with mutex (NOT recommended, serializes inference)
 *
 * See docs/THREAD_SAFETY.md for detailed examples and code samples.
 */
class VisionService {
public:
    /**
     * @brief Create a VisionService instance
     * @thread_safety Thread-safe. Can be called from multiple threads.
     */
    static std::unique_ptr<VisionService> Create(const std::string& config_path,
                                                const std::string& model_path_override = "",
                                                bool lazy_load = false);

    /**
     * @brief Get last creation error
     * @thread_safety Thread-safe (uses thread_local storage)
     */
    static const std::string& LastCreateError();

    /**
     * @brief Run inference on image file
     * @param conf_threshold Detection confidence threshold (<= 0 uses model default from config)
     * @param iou_threshold  NMS IoU threshold (<= 0 uses model default from config)
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     */
    VisionServiceStatus InferImage(const std::string& image_path,
                                    std::vector<VisionServiceResult>* out_results,
                                    float conf_threshold = -1.0f,
                                    float iou_threshold = -1.0f);

    /**
     * @brief Run inference on cv::Mat image
     * @param conf_threshold Detection confidence threshold (<= 0 uses model default from config)
     * @param iou_threshold  NMS IoU threshold (<= 0 uses model default from config)
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     */
    VisionServiceStatus InferImage(const cv::Mat& image,
                                    std::vector<VisionServiceResult>* out_results,
                                    float conf_threshold = -1.0f,
                                    float iou_threshold = -1.0f);

    /**
     * @brief Extract embedding from image file
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     */
    VisionServiceStatus InferEmbedding(const std::string& image_path,
                                        std::vector<float>* out_embedding);

    /**
     * @brief Extract embedding from cv::Mat image
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     */
    VisionServiceStatus InferEmbedding(const cv::Mat& image,
                                        std::vector<float>* out_embedding);

    /**
     * @brief Compute cosine similarity between two embeddings
     * @thread_safety Thread-safe (pure function)
     */
    static float EmbeddingSimilarity(const std::vector<float>& embedding_a,
                                    const std::vector<float>& embedding_b);

    /**
     * @brief Sequence action recognition (e.g. STGCN): 30-frame skeleton -> class probabilities
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     */
    VisionServiceStatus InferSequence(const float* pts, int image_width, int image_height,
                                        std::vector<float>* out_scores);

    /**
     * @brief Get class names for sequence model
     * @thread_safety Thread-safe after model is loaded
     */
    std::vector<std::string> GetSequenceClassNames();

    /**
     * @brief Get fall-down class index for STGCN (typically 6)
     * @thread_safety Thread-safe after model is loaded
     */
    int GetFallDownClassIndex();

    /**
     * @brief Draw inference results on image
     * @thread_safety NOT thread-safe. Do not call concurrently on the same instance.
     * @note Must call InferImage() first to have results to draw.
     */
    VisionServiceStatus Draw(const cv::Mat& image, cv::Mat* out_image);

    /**
     * @brief Check if current model supports drawing
     * @thread_safety Thread-safe after model is loaded
     */
    bool SupportsDraw() const;

    /**
     * @brief Release model resources
     * @thread_safety NOT thread-safe. Do not call while inference is running.
     */
    void Release();

    /**
     * @brief Set timing options
     * @thread_safety NOT thread-safe.
     */
    void SetTimingOptions(const VisionServiceTimingOptions& options);

    /**
     * @brief Get timing data from last inference
     * @thread_safety NOT thread-safe. Only call from the same thread that ran inference.
     */
    VisionServiceTiming GetLastTiming() const;

    /**
     * @brief Get default image path from config
     * @thread_safety Thread-safe after construction
     */
    std::string GetDefaultImage();

    /**
     * @brief Get config value by key
     * @thread_safety Thread-safe after construction
     */
    std::string GetConfigPathValue(const std::string& config_key);

    /**
     * @brief Get last error message
     * @thread_safety Thread-safe (uses instance-local storage)
     */
    const std::string& LastError() const;

    ~VisionService();

private:
    VisionService() = default;

    VisionServiceStatus SetError(VisionServiceStatus code, const std::string& message);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string last_error_;

    static thread_local std::string g_last_error_;
};

#endif  // VISION_SERVICE_H
