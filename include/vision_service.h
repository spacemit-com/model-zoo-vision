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

struct VisionServiceResult {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    int track_id;
};

struct VisionServiceKeypoint {
    float x;
    float y;
    float visibility;
};

enum VisionServiceStatus {
    VISION_SERVICE_OK = 0,
    VISION_SERVICE_INVALID_ARGUMENT = 1,
    VISION_SERVICE_CREATE_FAILED = 2,
    VISION_SERVICE_INFER_FAILED = 3,
    VISION_SERVICE_IO_FAILED = 4
};

class VisionService {
public:
    static std::unique_ptr<VisionService> Create(const std::string& config_path,
                                                const std::string& model_path_override = "",
                                                bool lazy_load = false);
    static const std::string& LastCreateError();

    VisionServiceStatus InferImage(const std::string& image_path,
                                    std::vector<VisionServiceResult>* out_results);
    VisionServiceStatus InferImage(const cv::Mat& image,
                                    std::vector<VisionServiceResult>* out_results);
    VisionServiceStatus InferEmbedding(const std::string& image_path,
                                        std::vector<float>* out_embedding);
    VisionServiceStatus InferEmbedding(const cv::Mat& image,
                                        std::vector<float>* out_embedding);
    static float EmbeddingSimilarity(const std::vector<float>& embedding_a,
                                    const std::vector<float>& embedding_b);

    /** Sequence action (e.g. STGCN): 30-frame skeleton -> class probabilities. */
    VisionServiceStatus InferSequence(const float* pts, int image_width, int image_height,
                                        std::vector<float>* out_scores);
    /** Class names for sequence model (e.g. STGCN). Empty if not a sequence model. */
    std::vector<std::string> GetSequenceClassNames();
    /** Fall-down class index for STGCN (typically 6). -1 if N/A. */
    int GetFallDownClassIndex();

    VisionServiceStatus Draw(const cv::Mat& image, cv::Mat* out_image);

    VisionServiceStatus GetLastKeypoints(int result_index,
                                            std::vector<VisionServiceKeypoint>* out_keypoints);

    std::string GetDefaultImage();
    std::string GetConfigPathValue(const std::string& config_key);

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
