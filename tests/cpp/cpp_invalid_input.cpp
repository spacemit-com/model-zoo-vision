/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PR invalid-input test: Create error paths only (no model required).
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include "vision_service.h"

namespace {

int g_failures = 0;

void fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    ++g_failures;
}

void check(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

std::string make_bad_yaml_path() {
    const char* out_dir = "tests/output";
    mkdir(out_dir, 0755);
    const std::string bad_yaml = std::string(out_dir) + "/bad_syntax.yaml";
    std::ofstream out(bad_yaml);
    out << ":\n  key: [unclosed\n";
    out.close();
    return bad_yaml;
}

void check_invalid_preprocess_config(
    const std::string& default_params,
    const std::string& expected_error) {
    const std::string path =
        "tests/output/invalid_preprocess_config.yaml";
    std::ofstream out(path);
    out << "class: deploy.yolov8.YOLOv8Detector\n"
        << "model_path: /no/such/model.onnx\n"
        << "default_params:\n"
        << default_params;
    out.close();

    auto service = VisionService::Create(path);
    check(service == nullptr, "expected invalid preprocess config to fail");
    const std::string error = VisionService::LastCreateError();
    check(
        error.find(expected_error) != std::string::npos,
        "expected '" + expected_error + "' in LastCreateError, got: " +
            error);
    std::remove(path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "Create(\"\")" << std::endl;
    auto empty_service = VisionService::Create("");
    check(empty_service == nullptr, "expected nullptr for empty config_path");
    {
        const std::string expected = "expected 'config_path is empty', got: ";
        const std::string msg = expected + VisionService::LastCreateError();
        check(VisionService::LastCreateError() == "config_path is empty", msg);
    }

    const std::string missing = "/no/such/config.yaml";
    std::cout << "Create(" << missing << ")" << std::endl;
    auto missing_service = VisionService::Create(missing);
    check(missing_service == nullptr, "expected nullptr for missing config");
    check(!VisionService::LastCreateError().empty(),
            "expected non-empty LastCreateError for missing config");
    {
        const std::string err = VisionService::LastCreateError();
        {
            const std::string msg =
                std::string("expected 'not found' in LastCreateError, got: ") + err;
            check(err.find("not found") != std::string::npos, msg);
        }
        {
            const std::string msg =
                std::string("expected path in LastCreateError, got: ") + err;
            check(err.find(missing) != std::string::npos, msg);
        }
    }

    const std::string bad_yaml = make_bad_yaml_path();
    std::cout << "Create(" << bad_yaml << ")" << std::endl;
    auto bad_service = VisionService::Create(bad_yaml);
    check(bad_service == nullptr, "expected nullptr for syntactically invalid yaml");
    check(!VisionService::LastCreateError().empty(),
            "expected non-empty LastCreateError for bad yaml, got empty");

    std::remove(bad_yaml.c_str());

    check_invalid_preprocess_config(
        "  - invalid\n",
        "default_params must be a map");
    check_invalid_preprocess_config(
        "  preprocess: opencl\n",
        "default_params.preprocess must be a map");
    check_invalid_preprocess_config(
        "  preprocess:\n"
        "    backend: [opencl]\n",
        "default_params.preprocess.backend must be a scalar");

    if (g_failures > 0) {
        std::cerr << g_failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: all Create error paths behaved as expected" << std::endl;
    return 0;
}
