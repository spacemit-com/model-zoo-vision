/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FACE_RECOGNITION_RUNTIME_H
#define FACE_RECOGNITION_RUNTIME_H

namespace face_recognition {

// Registration selects and aligns the best detected face before producing the
// single embedding that is persisted to the face database.
inline constexpr bool RunEmbeddingInRegistrationPipeline() {
    return false;
}

class CameraReadFailurePolicy {
public:
    explicit CameraReadFailurePolicy(int failure_limit) : failure_limit_(failure_limit) {}

    // Returns false once the configured number of consecutive failures is hit.
    bool OnReadFailure() {
        ++consecutive_failures_;
        return consecutive_failures_ < failure_limit_;
    }

    void OnReadSuccess() { consecutive_failures_ = 0; }

    int consecutive_failures() const { return consecutive_failures_; }

private:
    int failure_limit_;
    int consecutive_failures_ = 0;
};

}  // namespace face_recognition

#endif  // FACE_RECOGNITION_RUNTIME_H
