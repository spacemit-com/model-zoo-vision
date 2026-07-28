/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace {

class ProfileProbeModel final : public vision_core::BaseModel {
public:
    ProfileProbeModel() : BaseModel("unused", true) {}

    void load_model() override {}

    vision_core::InferResponse Run(const vision_core::InferRequest&) override {
        return {};
    }

    std::vector<vision_core::InferIntent> supported_intents() const override {
        return {vision_core::InferIntent::kDetect};
    }

    void Add(const std::string& name, double elapsed_ms, uint64_t calls = 1) {
        add_runtime_component_timing(name, elapsed_ms, calls);
    }
};

bool Near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

bool Check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
    }
    return condition;
}

}  // namespace

int main() {
    ProfileProbeModel model;
    model.Add("detector.infer", 3.0);
    model.Add("recognizer.infer", 4.0, 2);
    model.Add("detector.infer", 2.5);
    model.Add("zero.infer", 0.0);
    model.Add("", 8.0);
    model.Add("invalid.infer", -1.0);

    const auto profile = model.get_runtime_profile();
    if (!Check(profile.components.size() == 3, "invalid entries should be ignored") ||
        !Check(profile.components[0].name == "detector.infer", "first-seen order changed") ||
        !Check(Near(profile.components[0].total_ms, 5.5), "same-name time not accumulated") ||
        !Check(profile.components[0].calls == 2, "same-name calls not accumulated") ||
        !Check(profile.components[1].name == "recognizer.infer", "second entry changed") ||
        !Check(Near(profile.components[1].total_ms, 4.0), "recognizer time changed") ||
        !Check(profile.components[1].calls == 2, "explicit call count changed") ||
        !Check(profile.components[2].name == "zero.infer", "zero-time call was dropped") ||
        !Check(Near(profile.components[2].total_ms, 0.0), "zero-time total changed") ||
        !Check(profile.components[2].calls == 1, "zero-time call count changed")) {
        return 1;
    }

    model.reset_runtime_profile();
    if (!Check(
            model.get_runtime_profile().components.empty(),
            "reset did not clear components")) {
        return 1;
    }
    return 0;
}
