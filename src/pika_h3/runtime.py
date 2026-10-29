# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Reject an unqualified upstream implementation before loading model weights."""
import hashlib
import importlib.util
from pathlib import Path

EXPECTED = {
    "models/transformers/transformer_minimax_h3.py": "19e69741b2e03bf8b091603e815b02e963253239ee3085bd8ed2b9f6be044388",
    "schedulers/scheduling_minimax_h3.py": "c163bcbeeb4f05aa3eaf7708ebc551717b9d25d02a6ac9688228d7d0878d74ea",
}


def validate_runtime():
    spec = importlib.util.find_spec("diffusers")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install the pinned requirements first")
    root = Path(spec.origin).parent
    for name, checksum in EXPECTED.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError("Diffusers implementation mismatch; install requirements.txt and run python tools/apply_diffusers_patches.py")
