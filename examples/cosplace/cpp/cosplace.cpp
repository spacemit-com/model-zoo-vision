/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>    // NOLINT(build/include_order)
#include <iomanip>     // NOLINT(build/include_order)
#include <memory>      // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <variant>     // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)
#include <algorithm>   // NOLINT(build/include_order)
#include <cmath>       // NOLINT(build/include_order)
#include <stdexcept>   // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/c++17) NOLINT(build/include_order)

#include "vision_service.h"  // NOLINT(build/include_order)

struct PlaceMatch {
    int index;
    float distance;
    std::string path;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_yaml> [options]\n"
                << "  Options:\n"
                << "    --query <path>       Query image path (overrides test_query in config)\n"
                << "    --database <dir>     Database directory (overrides test_database in config)\n"
                << "    --top-k <n>          Number of top matches (overrides top_k in config, default: 5)\n"
                << "    --model-path <path>  Override model path from config\n"
                << "\n"
                << "  Example:\n"
                << "    " << argv[0] << " config/cosplace.yaml\n"
                << "    " << argv[0] << " config/cosplace.yaml --query query.jpg --database ./db_images --top-k 3\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string query_path;
    std::string database_dir;
    int top_k = -1;
    std::string model_path_override;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--query" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--database" && i + 1 < argc) {
            database_dir = argv[++i];
        } else if (arg == "--top-k" && i + 1 < argc) {
            try {
                top_k = std::stoi(argv[++i]);
                if (top_k <= 0) {
                    std::cerr << "Error: --top-k must be a positive integer" << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid --top-k value. Expected a positive integer." << std::endl;
                return 1;
            }
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path_override = argv[++i];
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

    // Read defaults from config if not provided via command line
    if (query_path.empty()) {
        query_path = service->GetConfigPathValue("test_query");
    }
    if (database_dir.empty()) {
        database_dir = service->GetConfigPathValue("test_database");
    }
    if (top_k == -1) {
        std::string top_k_str = service->GetConfigPathValue("top_k");
        top_k = top_k_str.empty() ? 5 : std::stoi(top_k_str);
    }

    if (query_path.empty()) {
        std::cerr << "Error: Query image required. Use --query <path> or set test_query in config" << std::endl;
        return 1;
    }

    if (database_dir.empty()) {
        std::cerr << "Error: Database directory required. Use --database <dir> or set test_database in config" << std::endl;
        return 1;
    }

    // Extract query embedding
    std::cout << "Extracting query embedding from: " << query_path << std::endl;
    VisionServiceResponse query_response;
    int ret = service->Infer(query_path, &query_response);
    if (ret != VISION_SERVICE_OK) {
        std::cerr << "Error inferring query: " << service->LastError() << std::endl;
        return 1;
    }

    if (query_response.results.empty()) {
        std::cerr << "Error: No embedding extracted from query image" << std::endl;
        return 1;
    }

    const vision::Embedding* query_emb = std::get_if<vision::Embedding>(&query_response.results[0]);
    if (!query_emb) {
        std::cerr << "Error: Expected Embedding result" << std::endl;
        return 1;
    }

    std::cout << "Query embedding dimension: " << query_emb->embedding.size() << std::endl;

    // Get database images using filesystem (avoids shell injection)
    std::vector<std::string> db_paths;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(database_dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    db_paths.push_back(entry.path().string());
                }
            }
        }
        std::sort(db_paths.begin(), db_paths.end());
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error reading database directory: " << e.what() << std::endl;
        return 1;
    }

    if (db_paths.empty()) {
        std::cerr << "Error: No images found in database directory: " << database_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << db_paths.size() << " images in database" << std::endl;
    std::cout << "Extracting database embeddings..." << std::endl;

    // Extract database embeddings (with valid path tracking)
    std::vector<std::vector<float>> db_embeddings;
    std::vector<std::string> valid_db_paths;
    db_embeddings.reserve(db_paths.size());
    valid_db_paths.reserve(db_paths.size());

    for (size_t i = 0; i < db_paths.size(); ++i) {
        VisionServiceResponse db_response;
        ret = service->Infer(db_paths[i], &db_response);
        if (ret != VISION_SERVICE_OK || db_response.results.empty()) {
            std::cerr << "Warning: Failed to extract embedding from " << db_paths[i] << std::endl;
            continue;
        }

        const vision::Embedding* db_emb = std::get_if<vision::Embedding>(&db_response.results[0]);
        if (db_emb) {
            db_embeddings.push_back(db_emb->embedding);
            valid_db_paths.push_back(db_paths[i]);
        }

        if ((i + 1) % 10 == 0 || i + 1 == db_paths.size()) {
            std::cout << "\r  Progress: " << (i + 1) << "/" << db_paths.size() << std::flush;
        }
    }
    std::cout << std::endl;

    if (db_embeddings.size() != db_paths.size()) {
        std::cerr << "Warning: Only extracted " << db_embeddings.size() << " of " << db_paths.size() << " embeddings" << std::endl;
    }

    // Compute L2 distances
    std::vector<PlaceMatch> matches;
    for (size_t i = 0; i < db_embeddings.size(); ++i) {
        float distance = 0.0f;
        for (size_t d = 0; d < query_emb->embedding.size(); ++d) {
            float diff = query_emb->embedding[d] - db_embeddings[i][d];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);

        PlaceMatch match;
        match.index = static_cast<int>(i);
        match.distance = distance;
        match.path = valid_db_paths[i];
        matches.push_back(match);
    }

    // Sort by distance (ascending)
    std::sort(matches.begin(), matches.end(),
            [](const PlaceMatch& a, const PlaceMatch& b) {
                return a.distance < b.distance;
            });

    // Display top-k results
    std::cout << "\n========================================" << std::endl;
    std::cout << "Top-" << top_k << " Place Matches (by L2 distance)" << std::endl;
    std::cout << "========================================" << std::endl;

    int display_count = std::min(top_k, static_cast<int>(matches.size()));
    for (int i = 0; i < display_count; ++i) {
        std::cout << "#" << (i + 1) << " Distance: " << std::fixed << std::setprecision(4)
                << matches[i].distance << " - " << matches[i].path << std::endl;
    }

    return 0;
}
