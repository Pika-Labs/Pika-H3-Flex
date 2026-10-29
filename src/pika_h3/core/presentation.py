# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""MiniMax-H3 prompt presentations, built with the pipeline's own tokenizer arithmetic.

A "presentation" is what MiniMax-H3 hands its Qwen3-VL conditioner: a `"<Picture i>: "` label plus a vision
block per visual condition, then the prompt verbatim — no chat template, no special tokens — together with
one MiniMax-H3 modality tag per token (text rows are tagged `1`, a vision block's rows `0`, the video tag).

The four tasks differ only in which images precede the prompt:

| task    | images                                        |
|---------|-----------------------------------------------|
| `t2va`  | none                                          |
| `i2va`  | the first frame of the clip, on the canvas    |
| `fl2va` | the first and last frames, on the canvas      |
| `l2va`  | the last frame of the clip, on the canvas     |
| `ref2va`| one mid-clip frame at the reference canvas    |

`MiniMaxH3FL2VATextEncoderStep.__call__` and `MiniMaxH3Ref2VATextEncoderStep._build_presentation` produce
byte-identical token streams for an image-only, audio-free presentation, so this module calls the latter
(a staticmethod, callable without a pipeline) for all four.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["TASK_IMAGE_SLOTS", "build_presentation", "vision_features"]

# Which frames each task's presentation shows, in packed order. `ref2va` is handled separately because its
# image is sized by the reference rule rather than put on the generated canvas.
TASK_IMAGE_SLOTS = {
    "t2va": (),
    "i2va": ("first",),
    "fl2va": ("first", "last"),
    "l2va": ("last",),
    "ref2va": ("reference",),
}


@dataclass(frozen=True)
class _ImageRef:
    """The only two attributes `_build_presentation` reads off a reference."""

    kind: str = "image"
    has_audio: bool = False


def vision_features(processor: Any, images: list) -> tuple[dict, list[int]]:
    """Run PIL images through Qwen3-VL's image processor, batched.

    Returns:
        `(vision_inputs, token_counts)` — the conditioner's own `pixel_values` / `image_grid_thw` kwargs,
        and the number of vision tokens each image's block occupies (`grid.prod() // merge_size ** 2`).
    """
    if not images:
        return {}, []
    features = processor.image_processor(images=images, return_tensors="pt")
    grid = features["image_grid_thw"]
    merge_size = processor.image_processor.merge_size**2
    vision_inputs = {"pixel_values": features["pixel_values"], "image_grid_thw": grid}
    return vision_inputs, [int(entry.prod()) // merge_size for entry in grid]


def build_presentation(
    tokenizer: Any,
    prompt: str,
    image_token_counts: list[int],
    *,
    text_tag: int = 1,
    video_tag: int = 0,
) -> tuple[list[int], list[int]]:
    """Tokenize a presentation of `len(image_token_counts)` image blocks followed by `prompt`.

    Returns:
        `(token_ids, token_tags)`, the same pair the pipeline's text-encoder blocks emit.
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    references = [_ImageRef() for _ in image_token_counts]
    return MiniMaxH3Ref2VATextEncoderStep._build_presentation(
        tokenizer,
        prompt,
        references,
        list(image_token_counts),
        [],
        [],
        text_tag=text_tag,
        video_tag=video_tag,
    )
