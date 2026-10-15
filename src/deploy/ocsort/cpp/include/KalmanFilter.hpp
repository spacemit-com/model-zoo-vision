/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KALMANFILTER_HPP
#define KALMANFILTER_HPP

#include <Eigen/Dense>  // NOLINT(build/include_order)
#include <optional>

namespace ocsort {

class KalmanFilterNew {
public:
    KalmanFilterNew();
    KalmanFilterNew(int dim_x_, int dim_z_);
    void predict();
    void update(const Eigen::VectorXf* measurement);

    int dim_z = 4;
    int dim_x = 7;
    Eigen::VectorXf x;
    Eigen::MatrixXf P;
    Eigen::MatrixXf Q;
    Eigen::Matrix<float, 7, 7> F;
    Eigen::Matrix<float, 4, 7> H;
    Eigen::Matrix<float, 4, 4> R;
    float _alpha_sq = 1.0f;
    Eigen::VectorXf z;
    Eigen::MatrixXf K;
    Eigen::VectorXf y;
    Eigen::MatrixXf S;
    const Eigen::MatrixXf I;
    Eigen::VectorXf x_prior;
    Eigen::MatrixXf P_prior;
    Eigen::VectorXf x_post;
    Eigen::MatrixXf P_post;
    bool observed = false;

private:
    struct State {
        Eigen::VectorXf x;
        Eigen::MatrixXf P;
    };
    void correct(const Eigen::VectorXf& measurement);
    void recover(const Eigen::VectorXf& measurement);

    // Recovery needs only the last real observation, its posterior, and the
    // number of missing frames. No pointers into caller storage or full
    // history.
    std::optional<Eigen::VectorXf> last_observation_;
    std::optional<State> frozen_state_;
    int missed_updates_ = 0;
};

}  // namespace ocsort

#endif  // KALMANFILTER_HPP
