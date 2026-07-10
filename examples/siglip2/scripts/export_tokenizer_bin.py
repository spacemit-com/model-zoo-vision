#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Maintainer tool: build tokenizer.bin for the SigLIP2 (Gemma) C++ tokenizer.

End users do NOT need this. tokenizer.bin is a data artifact hosted on the
model archive and pulled by scripts/download_models.sh, just like the ONNX
files. This script is only for (re)producing that artifact once from a
HuggingFace tokenizer.json, after which it should be uploaded to the archive.

Output format (must match src/deploy/siglip2/cpp/gemma_tokenizer.cpp):
    uint32  vocab_size                       (little-endian)
    for id in 0 .. vocab_size-1:
        uint16  token_len,  bytes token      (UTF-8, no terminator)
    uint32  merge_count
    for rank in 0 .. merge_count-1:
        uint16  len_a, bytes a               (left  piece)
        uint16  len_b, bytes b               (right piece)

Usage:
    python3 export_tokenizer_bin.py --hf-model google/siglip2-base-patch16-224 \\
        --output ~/.cache/models/vision/siglip2/tokenizer.bin

Requires tokenizer.json (fetch once):
    pip install transformers
    python3 -c "from transformers import AutoTokenizer; \\
        AutoTokenizer.from_pretrained('google/siglip2-base-patch16-224')"
"""
import argparse
import json
import os
import struct


def find_tokenizer_json(hf_model: str) -> str:
  import pathlib
  cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
  model_dir = "models--" + hf_model.replace("/", "--")
  candidates = list((cache / model_dir).rglob("tokenizer.json"))
  if not candidates:
    raise FileNotFoundError(
        f"tokenizer.json not found for {hf_model} in {cache}. "
        "Run: python3 -c \"from transformers import AutoTokenizer; "
        f"AutoTokenizer.from_pretrained('{hf_model}')\" first.")
  candidates.sort(key=lambda p: len(p.parts))
  return str(candidates[0])


def _merge_pair(entry):
  # HF stores merges either as ["a", "b"] pairs or as space-joined "a b".
  if isinstance(entry, (list, tuple)):
    return entry[0], entry[1]
  left, _, right = entry.partition(" ")
  return left, right


def _write_str(f, s: str):
  data = s.encode("utf-8")
  if len(data) > 0xFFFF:
    raise ValueError(f"Token too long for uint16 length: {repr(s)}")
  f.write(struct.pack("<H", len(data)))
  f.write(data)


def export(tokenizer_json_path: str, output_path: str):
  print(f"Loading: {tokenizer_json_path}")
  with open(tokenizer_json_path, encoding="utf-8") as f:
    t = json.load(f)

  model = t["model"]
  if model["type"] != "BPE":
    raise ValueError(f"Expected BPE tokenizer, got {model['type']}")

  vocab: dict[str, int] = model["vocab"]
  merges = model["merges"]

  vocab_size = len(vocab)
  id_to_token = [""] * vocab_size
  for token, idx in vocab.items():
    id_to_token[idx] = token

  print(f"  vocab_size:   {vocab_size}")
  print(f"  merge_count:  {len(merges)}")

  os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
  with open(output_path, "wb") as f:
    f.write(struct.pack("<I", vocab_size))
    for token_str in id_to_token:
      _write_str(f, token_str)
    f.write(struct.pack("<I", len(merges)))
    for entry in merges:
      a, b = _merge_pair(entry)
      _write_str(f, a)
      _write_str(f, b)

  size_mb = os.path.getsize(output_path) / 1024 / 1024
  print(f"[OK] Written {size_mb:.1f} MB -> {output_path}")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--tokenizer-json", help="Direct path to tokenizer.json")
  group.add_argument("--hf-model", help="HuggingFace model ID (auto-detect from cache)")
  parser.add_argument(
      "--output",
      default=os.path.expanduser("~/.cache/models/vision/siglip2/tokenizer.bin"),
  )
  args = parser.parse_args()

  path = args.tokenizer_json if args.tokenizer_json else find_tokenizer_json(args.hf_model)
  if not args.tokenizer_json:
    print(f"Found: {path}")
  export(path, args.output)


if __name__ == "__main__":
  main()
