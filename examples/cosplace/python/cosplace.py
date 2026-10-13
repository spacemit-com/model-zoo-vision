#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CosPlace visual place recognition example."""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from spacemit_vision import VisionServiceNative, VisionServiceStatus
except ImportError:
    print("Error: spacemit_vision is not installed.")
    print("Build the vision component and install the generated wheel "
          "(src/python/dist/*.whl), see the vision README section 2.1.")
    sys.exit(1)

DEFAULT_TOP_K = 5
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def positive_int(text):
    """argparse type: strictly positive integer."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {text}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got: {text}")
    return value


def get_image_paths(folder):
    """Recursively collect supported image files (case-insensitive extension)."""
    return sorted(
        str(p) for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_embedding(service, image_path):
    """Return the embedding of image_path as float32 array, or raise RuntimeError."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    status, embedding = service.infer_embedding(img)
    if status != VisionServiceStatus.OK:
        raise RuntimeError(f"inference failed: {service.last_error()}")
    if not embedding:
        raise RuntimeError(f"no embedding extracted from {image_path}")

    # infer_embedding returns a flat list of floats (the embedding vector)
    return np.asarray(embedding, dtype=np.float32)


def resolve_top_k(service, cli_value):
    """CLI value wins; otherwise read top_k from config. Returns None if invalid."""
    if cli_value is not None:
        return cli_value
    top_k_str = service.get_config_path_value("top_k")
    if not top_k_str:
        return DEFAULT_TOP_K
    try:
        return positive_int(top_k_str)
    except argparse.ArgumentTypeError:
        print(f"Error: Invalid 'top_k' in config. Expected a positive integer, got: {top_k_str}")
        return None


def main():
    parser = argparse.ArgumentParser(description='CosPlace Visual Place Recognition')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--query', type=str,
                        help='Path to query image (overrides test_query in config)')
    parser.add_argument('--database', type=str,
                        help='Path to database directory (overrides test_database in config)')
    parser.add_argument('--top-k', type=positive_int,
                        help=f'Number of top matches (overrides top_k in config, default: {DEFAULT_TOP_K})')
    parser.add_argument('--model-path', type=str,
                        help='Override model path from config')
    args = parser.parse_args()

    config_path = os.path.expanduser(args.config)

    print(f"Loading model from config: {config_path}")
    try:
        service = VisionServiceNative.create(config_path, args.model_path or "")
    except Exception as e:  # pybind raises ValueError/RuntimeError on create failure
        print(f"Error: Failed to create service: {e}")
        return 1
    if service is None:
        print(f"Error: Failed to create service from {config_path}")
        return 1

    # Command-line values take precedence over config defaults.
    query_path = args.query or service.get_config_path_value("test_query")
    database_dir = args.database or service.get_config_path_value("test_database")
    top_k = resolve_top_k(service, args.top_k)
    if top_k is None:
        return 1

    if not query_path:
        print("Error: Query image required. Use --query <path> or set test_query in config")
        return 1
    if not database_dir:
        print("Error: Database directory required. Use --database <dir> or set test_database in config")
        return 1

    query_path = os.path.expanduser(query_path)
    database_dir = os.path.expanduser(database_dir)
    if not os.path.isfile(query_path):
        print(f"Error: Query image not found: {query_path}")
        return 1
    if not os.path.isdir(database_dir):
        print(f"Error: Database directory not found: {database_dir}")
        return 1

    print(f"\nExtracting query embedding from: {query_path}")
    try:
        query_emb = extract_embedding(service, query_path)
    except RuntimeError as e:
        print(f"Error: Failed to extract query embedding: {e}")
        return 1
    query_dim = len(query_emb)
    print(f"Query embedding dimension: {query_dim}")

    db_paths = get_image_paths(database_dir)
    if not db_paths:
        print(f"Error: No images ({'/'.join(sorted(IMAGE_EXTENSIONS))}) found in database directory: "
              f"{database_dir}")
        return 1

    print(f"Found {len(db_paths)} images in database")
    print("Extracting database embeddings...")

    # Each match keeps its own path, so skipped images can never shift paths.
    matches = []
    for i, db_path in enumerate(db_paths):
        try:
            db_emb = extract_embedding(service, db_path)
        except RuntimeError as e:
            print(f"\nWarning: Failed to process {db_path}: {e}")
            db_emb = None

        if db_emb is not None:
            if len(db_emb) != query_dim:
                print(f"\nWarning: Skipping {db_path} "
                      f"(embedding dim {len(db_emb)} != query dim {query_dim})")
            else:
                dist = float(np.linalg.norm(query_emb - db_emb))
                matches.append((dist, db_path))

        if (i + 1) % 10 == 0 or i + 1 == len(db_paths):
            print(f"\r  Progress: {i + 1}/{len(db_paths)}", end='', flush=True)
    print()

    if len(matches) != len(db_paths):
        print(f"Warning: Only {len(matches)} of {len(db_paths)} database images were usable")
    if not matches:
        print("Error: No valid database embeddings to search")
        return 1

    matches.sort(key=lambda m: m[0])
    display_count = min(top_k, len(matches))

    print("\n" + "=" * 60)
    print(f"Top-{display_count} Place Matches (by L2 distance)")
    print("=" * 60)
    for rank, (dist, path) in enumerate(matches[:display_count], 1):
        print(f"#{rank} Distance: {dist:.4f} - {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
