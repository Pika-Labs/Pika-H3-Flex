# Third-party code notices

The inference and loader code is distributed under Apache-2.0 (LICENSE-CODE). The root LICENSE and NOTICE describe the MiniMax-H3-derived model adapters, not a replacement for Apache-2.0 on code.

## Diffusers

Pinned source: https://github.com/huggingface/diffusers/tree/3a2f35d4efa4c059c8bfb3bc0d6c906264895c81

The transformer and scheduler patches retain their original copyright and Apache-2.0 headers and identify Mellis modifications. Portions of geometry.py, packing.py and vae.py are adapted from the MiniMax H3 modular pipeline: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved. See licenses/Apache-2.0.txt.

## Model dependencies

Base model and Qwen3-VL encoder weights are downloaded separately at runtime; they are not bundled in this code repository or its Docker image. The downloader preserves available upstream LICENSE/NOTICE files. If encoder files are redistributed separately, retain their Apache-2.0 license and any applicable upstream NOTICE.
