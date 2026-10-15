/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/KalmanFilter.hpp"

#include <cmath>
#include <stdexcept>

namespace ocsort {

KalmanFilterNew::KalmanFilterNew() : KalmanFilterNew(7, 4) {}

KalmanFilterNew::KalmanFilterNew(int dim_x_, int dim_z_)
    : I(Eigen::MatrixXf::Identity(7, 7)) {
    if (dim_x_ != 7 || dim_z_ != 4) {
        throw std::invalid_argument(
            "OC-SORT requires a 7-state, 4-observation filter");
    }
    x = Eigen::VectorXf::Zero(dim_x);
    P = Eigen::MatrixXf::Identity(dim_x, dim_x);
    Q = P;
    F.setIdentity();
    H.setZero();
    R.setIdentity();
    z = Eigen::VectorXf::Zero(dim_z);
    K = Eigen::MatrixXf::Zero(dim_x, dim_z);
    y = Eigen::VectorXf::Zero(dim_z);
    S = Eigen::MatrixXf::Zero(dim_z, dim_z);
    x_prior = x;
    P_prior = P;
    x_post = x;
    P_post = P;
}

void KalmanFilterNew::predict() {
    x = (F * x).eval();
    P = (_alpha_sq * F * P * F.transpose() + Q).eval();
    x_prior = x;
    P_prior = P;
}

void KalmanFilterNew::correct(const Eigen::VectorXf& measurement) {
    y = measurement - H * x;
    const Eigen::MatrixXf pht = P * H.transpose();
    S = H * pht + R;
    const Eigen::LDLT<Eigen::MatrixXf> solver(S);
    if (solver.info() != Eigen::Success || !solver.isPositive()) {
        throw std::runtime_error(
            "OC-SORT innovation covariance is not positive definite");
    }
    K = solver.solve(pht.transpose()).transpose();
    x += K * y;
    const Eigen::MatrixXf residual = I - K * H;
    // Joseph form retains a symmetric, positive covariance under rounding.
    P = (residual * P * residual.transpose() + K * R * K.transpose()).eval();
    P = (0.5f * (P + P.transpose())).eval();
    z = measurement;
    x_post = x;
    P_post = P;
}

void KalmanFilterNew::recover(const Eigen::VectorXf& measurement) {
    const Eigen::VectorXf& previous = *last_observation_;
    if (previous[2] <= 0.0f || previous[3] <= 0.0f || measurement[2] <= 0.0f ||
        measurement[3] <= 0.0f) {
        throw std::invalid_argument(
            "OC-SORT recovery requires positive box area and aspect ratio");
    }
    const float width1 = std::sqrt(previous[2] * previous[3]);
    const float height1 = std::sqrt(previous[2] / previous[3]);
    const float width2 = std::sqrt(measurement[2] * measurement[3]);
    const float height2 = std::sqrt(measurement[2] / measurement[3]);
    const int gap = missed_updates_ + 1;
    x = frozen_state_->x;
    P = frozen_state_->P;
    // Rebuild each missed frame, ending with the current real measurement
    // exactly once. The caller's prediction for this frame is replaced here.
    for (int frame = 1; frame <= gap; ++frame) {
        predict();
        if (frame == gap) {
            correct(measurement);
        } else {
            const float alpha = static_cast<float>(frame) / gap;
            const float width = width1 + alpha * (width2 - width1);
            const float height = height1 + alpha * (height2 - height1);
            Eigen::VectorXf interpolated(4);
            interpolated << previous[0] +
                                alpha * (measurement[0] - previous[0]),
                previous[1] + alpha * (measurement[1] - previous[1]),
                width * height, width / height;
            correct(interpolated);
        }
    }
}

void KalmanFilterNew::update(const Eigen::VectorXf* measurement) {
    if (measurement == nullptr) {
        if (observed) {
            // x_post/P_post still belong to the last observed frame, even
            // though predict() has already advanced the current state.
            frozen_state_ = State{x_post, P_post};
        }
        observed = false;
        ++missed_updates_;
        z.setZero();
        y.setZero();
        x_post = x;
        P_post = P;
        return;
    }
    if (measurement->size() != dim_z || !measurement->allFinite()) {
        throw std::invalid_argument(
            "OC-SORT observation must contain four finite values");
    }
    if (frozen_state_ && last_observation_) {
        recover(*measurement);
    } else {
        correct(*measurement);
    }
    last_observation_ = *measurement;
    frozen_state_.reset();
    missed_updates_ = 0;
    observed = true;
}

}  // namespace ocsort
