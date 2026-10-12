#!/usr/bin/env python3
"""
CosPlace Visual Place Recognition Example
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    from spacemit_vision import VisionServiceNative, VisionServiceStatus
except ImportError:
    print("Error: spacemit_vision not installed. Please build and install the wheel:")
    print("  cd ../../ && cmake --build build -j && pip install src/python/dist/*.whl")
    sys.exit(1)


def get_image_paths(folder):
    """Recursively find all image files in folder"""
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    for ext in extensions:
        image_paths.extend(folder_path.rglob(f'*{ext}'))
        image_paths.extend(folder_path.rglob(f'*{ext.upper()}'))

    return sorted([str(p) for p in image_paths])


def extract_embedding(service, image_path):
    """Extract embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    status, results = service.infer_embedding(img)
    if status != VisionServiceStatus.OK:
        raise RuntimeError(f"Inference failed: {service.last_error()}")

    if not results or len(results) == 0:
        raise RuntimeError(f"No embedding extracted from {image_path}")

    # infer_embedding returns a flat list of floats (the embedding vector)
    return np.array(results, dtype=np.float32)


def compute_l2_distance(emb1, emb2):
    """Compute L2 distance between two embeddings"""
    diff = emb1 - emb2
    return np.sqrt(np.sum(diff * diff))


def main():
    parser = argparse.ArgumentParser(description='CosPlace Visual Place Recognition')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--query', type=str,
                        help='Path to query image (overrides test_query in config)')
    parser.add_argument('--database', type=str,
                        help='Path to database directory (overrides test_database in config)')
    parser.add_argument('--top-k', type=int,
                        help='Number of top matches (overrides top_k in config)')
    parser.add_argument('--model-path', type=str,
                        help='Override model path from config')

    args = parser.parse_args()

    # Expand user paths
    config_path = os.path.expanduser(args.config)

    # Create service
    print(f"Loading model from config: {config_path}")
    service = VisionServiceNative.create(config_path, args.model_path or "")

    # Read defaults from config if not provided via command line
    query_path = args.query if args.query else service.get_config_path_value("test_query")
    database_dir = args.database if args.database else service.get_config_path_value("test_database")
    if args.top_k is not None:
        top_k = args.top_k
    else:
        top_k_str = service.get_config_path_value("top_k")
        if not top_k_str:
            top_k = 5
        else:
            try:
                top_k = int(top_k_str)
                if top_k <= 0:
                    print(f"Error: Config 'top_k' must be a positive integer, got: {top_k_str}")
                    return 1
            except ValueError:
                print(f"Error: Invalid 'top_k' in config. Expected a positive integer, got: {top_k_str}")
                return 1

    if not query_path:
        print("Error: Query image required. Use --query <path> or set test_query in config")
        return 1

    if not database_dir:
        print("Error: Database directory required. Use --database <dir> or set test_database in config")
        return 1

    query_path = os.path.expanduser(query_path)
    database_dir = os.path.expanduser(database_dir)

    if not os.path.exists(query_path):
        print(f"Error: Query image not found: {query_path}")
        return 1

    if not os.path.isdir(database_dir):
        print(f"Error: Database directory not found: {database_dir}")
        return 1

    # Extract query embedding
    print(f"\nExtracting query embedding from: {query_path}")
    query_emb = extract_embedding(service, query_path)
    print(f"Query embedding dimension: {len(query_emb)}")

    # Get database images
    db_paths = get_image_paths(database_dir)
    if not db_paths:
        print(f"Error: No images found in database directory: {database_dir}")
        return 1

    print(f"Found {len(db_paths)} images in database")
    print("Extracting database embeddings...")

    # Extract database embeddings
    db_embeddings = []
    valid_db_paths = []

    for i, db_path in enumerate(db_paths):
        try:
            emb = extract_embedding(service, db_path)
            db_embeddings.append(emb)
            valid_db_paths.append(db_path)
        except Exception as e:
            print(f"\nWarning: Failed to process {db_path}: {e}")
            continue

        if (i + 1) % 10 == 0 or i + 1 == len(db_paths):
            print(f"\r  Progress: {i + 1}/{len(db_paths)}", end='', flush=True)

    print()

    if len(db_embeddings) != len(db_paths):
        print(f"Warning: Only extracted {len(db_embeddings)} embeddings")

    if not db_embeddings:
        print("Error: No valid embeddings extracted from database")
        return 1

    # Compute L2 distances
    distances = []
    for i, db_emb in enumerate(db_embeddings):
        dist = compute_l2_distance(query_emb, db_emb)
        distances.append((dist, i, valid_db_paths[i]))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[0])

    # Display top-k results
    print("\n" + "=" * 60)
    print(f"Top-{top_k} Place Matches (by L2 distance)")
    print("=" * 60)

    display_count = min(top_k, len(distances))
    for i in range(display_count):
        dist, idx, path = distances[i]
        print(f"#{i+1} Distance: {dist:.4f} - {path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
