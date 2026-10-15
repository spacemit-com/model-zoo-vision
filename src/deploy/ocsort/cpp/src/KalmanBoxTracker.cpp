/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/KalmanBoxTracker.hpp"

#include <stdexcept>
#include <utility>
namespace ocsort {
    KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXf bbox_, int cls_, int delta_t_, int id_) {
        if (bbox_.size() != 5 || !bbox_.allFinite() ||
            bbox_[2] <= bbox_[0] || bbox_[3] <= bbox_[1] || delta_t_ <= 0) {
            throw std::invalid_argument("OC-SORT requires a finite positive-area box and positive delta_t");
        }
        bbox = std::move(bbox_);
        delta_t = delta_t_;
        kf = std::make_unique<KalmanFilterNew>(7, 4);
        kf->F << 1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1;
        kf->H << 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
        kf->R.block(2, 2, 2, 2) *= 10.0;
        kf->P.block(4, 4, 3, 3) *= 1000.0;
        kf->P *= 10.0;
        kf->Q.bottomRightCorner(1, 1)(0, 0) *= 0.01;
        kf->Q.block(4, 4, 3, 3) *= 0.01;
        kf->x.head<4>() = convert_bbox_to_z(bbox);
        time_since_update = 0;
        id = id_;
        hits = 0;
        hit_streak = 0;
        age = 0;
        conf = bbox(4);
        cls = cls_;
        // Birth sets the initial state; only subsequent matches enter the
        // observation history and participate in direction estimation.
        last_observation.fill(-1);
        velocity.fill(0);
    }

    void KalmanBoxTracker::update(Eigen::Matrix<float, 5, 1>* bbox_, int cls_) {
        if (bbox_ != nullptr) {
            conf = (*bbox_)[4];
            cls = cls_;
            if (!observations.empty()) {
                const Eigen::VectorXf previous = k_previous_obs(observations, age, delta_t);
                velocity = speed_direction(previous, *bbox_);
            }
            last_observation = *bbox_;
            observations[age] = *bbox_;
            time_since_update = 0;
            hits += 1;
            hit_streak += 1;
            Eigen::VectorXf tmp = convert_bbox_to_z(*bbox_);
            kf->update(&tmp);
        } else {
            kf->update(nullptr);
        }
    }

    Eigen::RowVectorXf KalmanBoxTracker::predict() {
        if (kf->x[6] + kf->x[2] <= 0) kf->x[6] *= 0.0;
        kf->predict();
        age += 1;
        // Preserve the lookback window and, across long gaps, the last real
        // observation needed as fallback. Memory is independent of track age.
        while (observations.size() > 1 && observations.begin()->first < age - delta_t) {
            observations.erase(observations.begin());
        }
        if (time_since_update > 0) hit_streak = 0;
        time_since_update += 1;
        return convert_x_to_bbox(kf->x);
    }
    Eigen::VectorXf KalmanBoxTracker::get_state() {
        return convert_x_to_bbox(kf->x);
    }
}  // namespace ocsort
