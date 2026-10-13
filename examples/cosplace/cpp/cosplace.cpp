/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>   // NOLINT(build/include_order)
#include <cctype>      // NOLINT(build/include_order)
#include <cmath>       // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <iostream>    // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <stdexcept>   // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <variant>     // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

namespace fs = std::filesystem;

namespace {

constexpr int kDefaultTopK = 5;

struct PlaceMatch {
    float distance;
    std::string path;
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <config_yaml> [options]\n"
            << "  Options:\n"
            << "    --query <path>       Query image path (overrides test_query in config)\n"
            << "    --database <dir>     Database directory (overrides test_database in config)\n"
            << "    --top-k <n>          Number of top matches (overrides top_k in config, default: 5)\n"
            << "    --model-path <path>  Override model path from config\n"
            << "\n"
            << "  Supported database image extensions: .jpg .jpeg .png\n"
            << "\n"
            << "  Example:\n"
            << "    " << prog << " config/cosplace.yaml\n"
            << "    " << prog << " config/cosplace.yaml --query query.jpg --database ./db_images --top-k 3\n";
}

// Parses a strictly positive integer; rejects trailing garbage such as "3abc".
bool ParsePositiveInt(const std::string& text, int* value) {
    try {
        size_t pos = 0;
        const int parsed = std::stoi(text, &pos);
        if (pos != text.size() || parsed <= 0) {
            return false;
        }
        *value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool IsSupportedImage(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

// Returns false (with a message on stderr) if the embedding cannot be extracted.
bool ExtractEmbedding(VisionService* service, const std::string& path, std::vector<float>* out) {
    VisionServiceResponse response;
    if (service->Infer(path, &response) != VISION_SERVICE_OK) {
        std::cerr << "Warning: Inference failed for " << path << ": " << service->LastError() << std::endl;
        return false;
    }
    if (response.results.empty()) {
        std::cerr << "Warning: No result for " << path << std::endl;
        return false;
    }
    const vision::Embedding* emb = std::get_if<vision::Embedding>(&response.results[0]);
    if (emb == nullptr || emb->embedding.empty()) {
        std::cerr << "Warning: No embedding in result for " << path << std::endl;
        return false;
    }
    *out = emb->embedding;
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string config_path = argv[1];
    std::string query_path;
    std::string database_dir;
    int top_k = -1;
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--query" || arg == "--database" || arg == "--top-k" || arg == "--model-path") {
            if (!has_value) {
                std::cerr << "Error: " << arg << " requires a value" << std::endl;
                return 1;
            }
            const std::string value = argv[++i];
            if (arg == "--query") {
                query_path = value;
            } else if (arg == "--database") {
                database_dir = value;
            } else if (arg == "--model-path") {
                model_path_override = value;
            } else if (!ParsePositiveInt(value, &top_k)) {
                std::cerr << "Error: Invalid --top-k value '" << value
                        << "'. Expected a positive integer." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    std::unique_ptr<VisionService> service = VisionService::Create(
        config_path,
        model_path_override,
        true);
    if (!service) {
        std::cerr << "Error: " << VisionService::LastCreateError() << std::endl;
        return 1;
    }

    // Command-line values take precedence over config defaults.
    if (query_path.empty()) {
        query_path = service->GetConfigPathValue("test_query");
    }
    if (database_dir.empty()) {
        database_dir = service->GetConfigPathValue("test_database");
    }
    if (top_k == -1) {
        const std::string top_k_str = service->GetConfigPathValue("top_k");
        if (top_k_str.empty()) {
            top_k = kDefaultTopK;
        } else if (!ParsePositiveInt(top_k_str, &top_k)) {
            std::cerr << "Error: Invalid 'top_k' in config. Expected a positive integer, got: "
                    << top_k_str << std::endl;
            return 1;
        }
    }

    if (query_path.empty()) {
        std::cerr << "Error: Query image required. Use --query <path> or set test_query in config" << std::endl;
        return 1;
    }
    if (database_dir.empty()) {
        std::cerr << "Error: Database directory required. Use --database <dir> or set test_database in config"
                << std::endl;
        return 1;
    }

    std::error_code ec;
    if (!fs::is_regular_file(query_path, ec)) {
        std::cerr << "Error: Query image not found: " << query_path << std::endl;
        return 1;
    }
    if (!fs::is_directory(database_dir, ec)) {
        std::cerr << "Error: Database directory not found: " << database_dir << std::endl;
        return 1;
    }

    // Extract query embedding
    std::cout << "Extracting query embedding from: " << query_path << std::endl;
    std::vector<float> query_emb;
    if (!ExtractEmbedding(service.get(), query_path, &query_emb)) {
        std::cerr << "Error: Failed to extract query embedding" << std::endl;
        return 1;
    }
    const size_t query_dim = query_emb.size();
    std::cout << "Query embedding dimension: " << query_dim << std::endl;

    // Collect database images with std::filesystem (no shell involved).
    std::vector<std::string> db_paths;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(
                database_dir, fs::directory_options::skip_permission_denied)) {
            if (entry.is_regular_file() && IsSupportedImage(entry.path())) {
                db_paths.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error reading database directory: " << e.what() << std::endl;
        return 1;
    }
    std::sort(db_paths.begin(), db_paths.end());

    if (db_paths.empty()) {
        std::cerr << "Error: No images (.jpg/.jpeg/.png) found in database directory: " << database_dir
                << std::endl;
        return 1;
    }

    std::cout << "Found " << db_paths.size() << " images in database" << std::endl;
    std::cout << "Extracting database embeddings..." << std::endl;

    // Each match keeps its own path, so skipped images can never shift paths.
    std::vector<PlaceMatch> matches;
    matches.reserve(db_paths.size());
    for (size_t i = 0; i < db_paths.size(); ++i) {
        std::vector<float> db_emb;
        if (ExtractEmbedding(service.get(), db_paths[i], &db_emb)) {
            if (db_emb.size() != query_dim) {
                std::cerr << "Warning: Skipping " << db_paths[i] << " (embedding dim " << db_emb.size()
                        << " != query dim " << query_dim << ")" << std::endl;
            } else {
                float sum = 0.0f;
                for (size_t d = 0; d < query_dim; ++d) {
                    const float diff = query_emb[d] - db_emb[d];
                    sum += diff * diff;
                }
                matches.push_back({std::sqrt(sum), db_paths[i]});
            }
        }

        if ((i + 1) % 10 == 0 || i + 1 == db_paths.size()) {
            std::cout << "\r  Progress: " << (i + 1) << "/" << db_paths.size() << std::flush;
        }
    }
    std::cout << std::endl;

    if (matches.size() != db_paths.size()) {
        std::cerr << "Warning: Only " << matches.size() << " of " << db_paths.size()
                << " database images were usable" << std::endl;
    }
    if (matches.empty()) {
        std::cerr << "Error: No valid database embeddings to search" << std::endl;
        return 1;
    }

    // Sort by distance (ascending)
    std::sort(matches.begin(), matches.end(),
            [](const PlaceMatch& a, const PlaceMatch& b) {
                return a.distance < b.distance;
            });

    const int display_count = std::min(top_k, static_cast<int>(matches.size()));
    std::cout << "\n========================================" << std::endl;
    std::cout << "Top-" << display_count << " Place Matches (by L2 distance)" << std::endl;
    std::cout << "========================================" << std::endl;
    for (int i = 0; i < display_count; ++i) {
        std::cout << "#" << (i + 1) << " Distance: " << std::fixed << std::setprecision(4)
                << matches[i].distance << " - " << matches[i].path << std::endl;
    }

    return 0;
}
