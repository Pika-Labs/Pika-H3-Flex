# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Public generation command; validate inputs before loading models."""
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(prog="pika-h3-pro")
    p.add_argument("--task", choices=["t2va", "i2va", "l2va", "fl2va", "ref2va"], required=True)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    p.add_argument("--first-image", type=Path)
    p.add_argument("--last-image", type=Path)
    p.add_argument("--ref", action="append", default=[], help="Ordered image:path, video:path or audio:path reference")
    p.add_argument("--resolution", choices=["480p", "768p"], default="768p")
    p.add_argument("--aspect", choices=["21:9","16:9","3:2","4:3","5:4","1:1","4:5","3:4","2:3","9:16","9:21"], default="16:9")
    p.add_argument("--duration", type=int, choices=range(5,16), default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-offload", action="store_true", help="Offload transformer blocks to CPU between forwards to reduce VRAM; slower and requires host RAM")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-repo", default="Pika-Labs/Pika-H3-Flex")
    p.add_argument("--revision", required=True, help="Immutable HF adapter repository commit")
    p.add_argument("--base-path", type=Path, help="Optional complete local base snapshot")
    p.add_argument("--text-encoder-path", type=Path, help="Optional complete local conditioner snapshot")
    p.add_argument("--adapter-path", type=Path, help="Optional local exported adapter directory")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if not args.prompt or not args.prompt.strip() or len(args.prompt) > 8000:
        p.error("prompt must contain 1–8000 characters")
    if not 0 <= args.seed < 2**63:
        p.error("seed must be in [0, 2**63-1]")
    expected = {"t2va": (False,False), "i2va": (True,False), "l2va": (False,True), "fl2va": (True,True), "ref2va": (False,False)}[args.task]
    if (args.first_image is not None, args.last_image is not None) != expected:
        p.error("first/last images must match the selected task")
    if (args.task == "ref2va") != bool(args.ref):
        p.error("references are required exclusively for ref2va")
    if args.output.suffix.lower() != ".mp4":
        p.error("output must have an .mp4 extension")
    import re
    if not args.adapter_path and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        p.error("revision must be the full immutable Hugging Face commit SHA")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        p.error("output or metadata file already exists; choose a new output path")
    from .pipeline import generate
    generate(args)


if __name__ == "__main__":
    main()
