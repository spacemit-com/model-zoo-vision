/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <opencv2/opencv.hpp>

#include "benchmark_scenario.h"
#include "benchmark_stats.h"
#include "vision_service.h"

namespace {

struct Args {
    std::string config_path;
    std::string input_path;
    std::string input_path2;
    bool enable_similarity = false;
    std::string model_path_override;
    int runs = 100;
    int warmup = 5;
    bool measure_draw = false;
    bool verbose_timing = false;
};

void print_usage(const char *prog) {
    std::cout << "Usage: " << prog
                << " --config <yaml> [--image <path>] [--image2 <path>] "
                << "[--model-path <path>] [--runs N] [--warmup N] "
                << "[--measure-draw] [--verbose-timing]\n"
                << "Example: " << prog
                << " --config examples/yolov8/config/yolov8.yaml\n"
                << "The input kind is selected by config benchmark.scenario. "
                << "--image is the primary image/video override and --image2 is "
                << "the secondary stereo/matching image or embedding comparison "
                << "override.\n";
}

bool parse_args(int argc, char **argv, Args &args) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--config" && i + 1 < argc) {
                args.config_path = argv[++i];
            } else if (argument == "--image" && i + 1 < argc) {
                args.input_path = argv[++i];
            } else if (argument == "--image2" && i + 1 < argc) {
                args.input_path2 = argv[++i];
                args.enable_similarity = true;
            } else if (argument == "--model-path" && i + 1 < argc) {
                args.model_path_override = argv[++i];
            } else if (argument == "--runs" && i + 1 < argc) {
                args.runs = std::max(1, std::stoi(argv[++i]));
            } else if (argument == "--warmup" && i + 1 < argc) {
                args.warmup = std::max(0, std::stoi(argv[++i]));
            } else if (argument == "--measure-draw") {
                args.measure_draw = true;
            } else if (argument == "--verbose-timing") {
                args.verbose_timing = true;
            } else if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return false;
            } else {
                std::cerr << "Unknown or incomplete argument: " << argument
                            << std::endl;
                print_usage(argv[0]);
                return false;
            }
        }
    } catch (const std::exception &exception) {
        std::cerr << "Invalid argument: " << exception.what() << std::endl;
        return false;
    }

    if (args.config_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

double to_ms(const std::chrono::steady_clock::duration &duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

bool is_embedding_response(const VisionServiceResponse &response) {
    return !response.results.empty() &&
            std::holds_alternative<vision::Embedding>(response.results.front());
}

bool prepare_request(vision_benchmark::BenchmarkScenario *scenario,
                    VisionServiceRequest *request, const char *phase,
                    int index) {
    std::string error;
    if (scenario->NextRequest(request, &error)) {
        return true;
    }
    std::cerr << "Error: " << phase << " input " << index
                << " failed: " << error << std::endl;
    return false;
}

} // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    auto service = VisionService::Create(args.config_path,
                                        args.model_path_override, false);
    if (!service) {
        std::cerr << "Error: create service failed: "
                    << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    vision_benchmark::ScenarioOverrides overrides;
    overrides.input_path = args.input_path;
    overrides.input_path2 = args.input_path2;
    std::string scenario_error;
    auto scenario = vision_benchmark::CreateBenchmarkScenario(
        args.config_path, overrides, service.get(), &scenario_error);
    if (!scenario) {
        std::cerr << "Error: prepare benchmark scenario failed: "
                    << scenario_error << std::endl;
        return 1;
    }

    if (!scenario->Initialize(service.get(), &scenario_error)) {
        std::cerr << "Error: initialize benchmark scenario failed: "
                    << scenario_error << std::endl;
        return 1;
    }
    // Enable instrumentation only after scenario setup/initialization, so
    // warmup and measured samples cannot inherit setup timing.
    VisionServiceTimingOptions timing_options;
    timing_options.enabled = true;
    // Service printing occurs inside Infer(); defer diagnostic output until
    // all samples have finished so console I/O cannot inflate wall latency.
    timing_options.print_to_stdout = false;
    service->SetTimingOptions(timing_options);

    VisionServiceResponse response;
    bool embedding_mode = false;
    bool response_kind_known = false;
    for (int index = 0; index < args.warmup; ++index) {
        VisionServiceRequest request;
        if (!prepare_request(scenario.get(), &request, "warmup", index + 1)) {
            return 1;
        }
        if (service->Infer(request, &response) != VISION_SERVICE_OK) {
            std::cerr << "Error: warmup Infer failed at sample " << index + 1
                        << ": " << service->LastError() << std::endl;
            return 1;
        }
        if (!response_kind_known) {
            embedding_mode = is_embedding_response(response);
            response_kind_known = true;
        }
    }

    std::vector<double> infer_samples;
    infer_samples.reserve(static_cast<std::size_t>(args.runs));
    std::vector<VisionServiceTiming> verbose_samples;
    if (args.verbose_timing) {
        verbose_samples.reserve(static_cast<std::size_t>(args.runs));
    }
    double draw_total = 0.0;
    double service_preprocess_total = 0.0;
    double service_model_infer_total = 0.0;
    double service_postprocess_total = 0.0;
    double service_detect_total = 0.0;
    double service_track_total = 0.0;
    int draw_skipped_runs = 0;
    int draw_executed_runs = 0;
    const bool draw_supported = service->SupportsDraw();
    vision_benchmark::ComponentTimingAccumulator component_stats;
    cv::Mat drawn;

    for (int index = 0; index < args.runs; ++index) {
        VisionServiceRequest request;
        if (!prepare_request(scenario.get(), &request, "measured", index + 1)) {
            return 1;
        }

        const auto infer_begin = std::chrono::steady_clock::now();
        const VisionServiceStatus status = service->Infer(request, &response);
        const auto infer_end = std::chrono::steady_clock::now();
        if (status != VISION_SERVICE_OK) {
            std::cerr << "Error: Infer failed at measured sample " << index + 1
                        << ": " << service->LastError() << std::endl;
            return 1;
        }
        infer_samples.push_back(to_ms(infer_end - infer_begin));

        if (!response_kind_known) {
            embedding_mode = is_embedding_response(response);
            response_kind_known = true;
        }

        const VisionServiceTiming timing = service->GetLastTiming();
        if (args.verbose_timing) {
            verbose_samples.push_back(timing);
        }
        service_preprocess_total += timing.preprocess_ms;
        service_model_infer_total += timing.model_infer_ms;
        service_postprocess_total += timing.postprocess_ms;
        service_detect_total += timing.detect_ms;
        service_track_total += timing.track_ms;
        component_stats.Add(service->GetLastProfile().components);

        const cv::Mat &draw_image = scenario->draw_image();
        const bool skip_draw = !args.measure_draw || embedding_mode ||
                                response.results.empty() || !draw_supported ||
                                draw_image.empty();
        if (skip_draw) {
            ++draw_skipped_runs;
            continue;
        }

        const auto draw_begin = std::chrono::steady_clock::now();
        const VisionServiceStatus draw_status =
            service->Draw(draw_image, response, &drawn);
        const auto draw_end = std::chrono::steady_clock::now();
        if (draw_status != VISION_SERVICE_OK) {
            std::cerr << "Error: Draw failed at sample " << index + 1 << ": "
                        << service->LastError() << std::endl;
            return 1;
        }
        ++draw_executed_runs;
        draw_total += to_ms(draw_end - draw_begin);
    }

    for (std::size_t index = 0; index < verbose_samples.size(); ++index) {
        const auto &timing = verbose_samples[index];
        const double profile_total = std::max(
            {timing.infer_ms, timing.embedding_ms, timing.sequence_ms});
        std::cout << "Sample " << index + 1 << ": infer="
                    << infer_samples[index] << " preprocess=" << timing.preprocess_ms
                    << " model=" << timing.model_infer_ms
                    << " postprocess=" << timing.postprocess_ms
                    << " profile_total=" << profile_total
                    << " track=" << timing.track_ms << " ms\n";
    }

    const double infer_total =
        std::accumulate(infer_samples.begin(), infer_samples.end(), 0.0);
    const double avg_infer = infer_total / static_cast<double>(args.runs);
    const double avg_service_preprocess =
        service_preprocess_total / static_cast<double>(args.runs);
    const double avg_service_model_infer =
        service_model_infer_total / static_cast<double>(args.runs);
    const double avg_service_postprocess =
        service_postprocess_total / static_cast<double>(args.runs);
    const double avg_service_detect =
        service_detect_total / static_cast<double>(args.runs);
    const double avg_service_track =
        service_track_total / static_cast<double>(args.runs);
    const double avg_draw =
        draw_executed_runs > 0
            ? draw_total / static_cast<double>(draw_executed_runs)
            : 0.0;
    const double fps = avg_infer > 0.0 ? 1000.0 / avg_infer : 0.0;
    const bool looks_like_tracking =
        !embedding_mode &&
        (avg_service_detect > 0.0 || avg_service_track > 0.0);
    // Detection overlaps the detector's three stages, but tracking does not.
    // Keep the residual additive with the printed stages while retaining
    // service dispatch, result cleanup, and model-timer gaps in "other".
    const double avg_other = avg_infer - avg_service_preprocess -
                            avg_service_model_infer - avg_service_postprocess -
                            (looks_like_tracking ? avg_service_track : 0.0);

    std::cout << "Mode: " << (embedding_mode ? "InferEmbedding" : "InferImage")
                << "\n"
                << "Scenario: " << scenario->name() << "\n"
                << "Scope: " << scenario->measured_scope() << "\n"
                << "Config: " << args.config_path << "\n"
                << "Input: " << scenario->input_description() << "\n"
                << "Runs: " << args.runs << ", warmup " << args.warmup << "\n"
                << "Avg infer: " << avg_infer << " ms\n"
                << "Infer latency min/p50/p90/p95/max: "
                << *std::min_element(infer_samples.begin(), infer_samples.end())
                << "/" << percentile(infer_samples, 0.50) << "/"
                << percentile(infer_samples, 0.90) << "/"
                << percentile(infer_samples, 0.95) << "/"
                << *std::max_element(infer_samples.begin(), infer_samples.end())
                << " ms\n";
    vision_benchmark::PrintBenchmarkTimingSummary(
        std::cout, avg_service_preprocess, avg_service_model_infer,
        avg_service_postprocess, looks_like_tracking, avg_service_detect,
        avg_service_track);
    auto component_averages = component_stats.Averages(args.runs);
    const auto find_component = [&](const char *name)
        -> const vision_benchmark::ComponentTimingAverage * {
        const auto found = std::find_if(
            component_averages.begin(), component_averages.end(),
            [name](const auto &entry) { return entry.name == name; });
        return found == component_averages.end() ? nullptr : &*found;
    };
    const auto *ocr_detect = find_component("ocr.detect");
    const auto *ocr_recognize = find_component("ocr.recognize");
    if (ocr_detect != nullptr && ocr_recognize != nullptr) {
        std::cout << "Avg detect: " << ocr_detect->ms_per_run << " ms\n"
                    << "Avg recognize: " << ocr_recognize->ms_per_run << " ms\n";
    }
    std::cout << "Avg other (outside stage timers): " << avg_other
                << " ms\n";
    if (ocr_detect != nullptr && ocr_recognize != nullptr) {
        const auto component_ms = [&](const char *name) {
            const auto *entry = find_component(name);
            return entry == nullptr ? 0.0 : entry->ms_per_run;
        };
        std::ostringstream table;
        table << std::fixed << std::setprecision(3)
                << "Model components (ms/run):\n"
                << "  " << std::left << std::setw(12) << "Stage"
                << std::right << std::setw(12) << "Preprocess"
                << std::setw(13) << "Model infer"
                << std::setw(14) << "Postprocess"
                << std::setw(12) << "Calls/run" << '\n';
        const auto print_ocr_row = [&](const char *label, const char *prefix) {
            const std::string component_prefix(prefix);
            const auto *infer = find_component(
                (component_prefix + ".infer").c_str());
            table << "  " << std::left << std::setw(12) << label
                    << std::right << std::setw(12)
                    << component_ms((component_prefix + ".preprocess").c_str())
                    << std::setw(13) << component_ms(
                        (component_prefix + ".infer").c_str())
                    << std::setw(14) << component_ms(
                        (component_prefix + ".postprocess").c_str())
                    << std::defaultfloat << std::setprecision(4)
                    << std::setw(12)
                    << (infer == nullptr ? 0.0 : infer->calls_per_run)
                    << std::fixed << std::setprecision(3) << '\n';
        };
        print_ocr_row("Detect", "detector");
        print_ocr_row("Recognize", "recognizer");
        const auto *recognizer_infer = find_component("recognizer.infer");
        if (recognizer_infer != nullptr) {
            table << "  recognizer.infer: "
                    << recognizer_infer->ms_per_call << " ms/call\n";
        }
        for (const auto &entry : component_averages) {
            if (entry.name.rfind("image_preprocess.", 0) == 0) {
                table << "  " << entry.name << ": " << entry.ms_per_run
                        << " ms/run (part of Detect preprocess)\n";
            } else if (entry.name != "ocr.detect" &&
                        entry.name != "ocr.recognize" &&
                        entry.name != "detector.preprocess" &&
                        entry.name != "detector.infer" &&
                        entry.name != "detector.postprocess" &&
                        entry.name != "recognizer.preprocess" &&
                        entry.name != "recognizer.infer" &&
                        entry.name != "recognizer.postprocess") {
                table << "  " << entry.name << ": " << entry.ms_per_run
                        << " ms/run, " << entry.calls_per_run
                        << " calls/run, " << entry.ms_per_call
                        << " ms/call\n";
            }
        }
        std::cout << table.str();
    } else {
        vision_benchmark::PrintComponentTimings(
            std::cout, component_averages);
    }
    std::cout << "Draw measurement: "
                << (args.measure_draw ? "enabled" : "disabled") << "\n"
                << "Draw executed/skipped: " << draw_executed_runs << "/"
                << draw_skipped_runs << "\n"
                << "Avg draw: " << avg_draw << " ms\n"
                << "FPS: " << fps << "\n";

    if (embedding_mode && args.enable_similarity && !args.input_path2.empty()) {
        cv::Mat image2 = cv::imread(args.input_path2);
        if (image2.empty()) {
            std::cerr << "Warning: could not read image2 for similarity: "
                        << args.input_path2 << std::endl;
        } else {
            std::vector<float> embedding;
            if (!response.results.empty()) {
                if (const auto *value = std::get_if<vision::Embedding>(
                        &response.results.front())) {
                    embedding = value->embedding;
                }
            }
            VisionServiceResponse response2;
            const VisionServiceStatus status2 =
                service->Infer(image2, &response2);
            std::vector<float> embedding2;
            if (status2 == VISION_SERVICE_OK && !response2.results.empty()) {
                if (const auto *value = std::get_if<vision::Embedding>(
                        &response2.results.front())) {
                    embedding2 = value->embedding;
                }
            }
            if (status2 == VISION_SERVICE_OK && !embedding.empty() &&
                !embedding2.empty()) {
                std::cout << "Image2: " << args.input_path2 << "\n"
                            << "Embedding similarity: "
                            << VisionService::EmbeddingSimilarity(embedding,
                                                                embedding2)
                            << "\n";
            } else {
                std::cerr << "Warning: failed to infer image2 embedding: "
                            << service->LastError() << std::endl;
            }
        }
    }

    return 0;
}
