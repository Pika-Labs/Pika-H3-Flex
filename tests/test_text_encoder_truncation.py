# SPDX-License-Identifier: Apache-2.0
"""CPU check of the text-encoder truncation: a small randomly initialised Qwen3-VL gives a bit-identical
``hidden_states[layer]`` with and without the dropped layers.

    PYTHONPATH=src python -m unittest tests.test_text_encoder_truncation -v
"""
import copy
import os
import unittest

import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from pika_h3.core.conditioning import TEXT_ENCODER_LAYER, truncate_text_encoder


def tiny_qwen3vl(num_layers):
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=num_layers, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[]),
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def hidden_state(te, input_ids, layer):
    # the same call as diffusers' get_qwen3vl_prompt_embeds (text-only presentation)
    out = te.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, output_hidden_states=True)
    return out.hidden_states[layer]


class TextEncoderTruncation(unittest.TestCase):
    def check(self, num_layers, layer, dtype):
        full = tiny_qwen3vl(num_layers).to(dtype)
        trunc = copy.deepcopy(full)
        self.assertEqual(truncate_text_encoder(trunc, layer), min(num_layers, layer + 1))
        ids = torch.randint(0, 256, (1, 37), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            a, b = hidden_state(full, ids, layer), hidden_state(trunc, ids, layer)
        self.assertTrue(torch.equal(a, b), f"hidden_states[{layer}] differs after truncation ({dtype})")
        # the read entry is NOT the post-norm last one
        with torch.no_grad():
            last = trunc.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True).hidden_states
        self.assertEqual(len(last), layer + 2)

    def test_bit_identical_fp32(self):
        self.check(8, 5, torch.float32)

    def test_bit_identical_bf16(self):
        self.check(8, 5, torch.bfloat16)

    def test_released_layer_index(self):
        # the released conditioner: 64 layers, hidden_states[50] -> 51 layers kept
        self.check(TEXT_ENCODER_LAYER + 14, TEXT_ENCODER_LAYER, torch.float32)

    def test_opt_out(self):
        te = tiny_qwen3vl(8)
        os.environ["PIKA_TE_TRUNC"] = "0"
        try:
            self.assertEqual(truncate_text_encoder(te, 5), 8)
        finally:
            del os.environ["PIKA_TE_TRUNC"]


if __name__ == "__main__":
    unittest.main()
