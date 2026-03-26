import unittest

import numpy as np
from transformers import AutoConfig

from common.rope import apply_rope, relative_kernel, rope_cos_sin, rope_metadata_from_config


class RopeUtilsTests(unittest.TestCase):
    def test_gpt2_has_no_rope_metadata(self):
        cfg = AutoConfig.from_pretrained('gpt2')
        self.assertIsNone(rope_metadata_from_config(cfg))

    def test_pythia_rope_metadata_uses_partial_rotary_factor(self):
        cfg = AutoConfig.from_pretrained('EleutherAI/pythia-70m')
        meta = rope_metadata_from_config(cfg)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.head_dim, 64)
        self.assertEqual(meta.rotary_dim, 16)
        self.assertEqual(meta.num_attention_heads, 8)
        self.assertEqual(meta.num_key_value_heads, 8)

    def test_llama_family_rope_metadata_uses_full_head_dim(self):
        cfg = AutoConfig.from_pretrained('HuggingFaceTB/SmolLM2-135M')
        meta = rope_metadata_from_config(cfg)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.head_dim, 64)
        self.assertEqual(meta.rotary_dim, 64)
        self.assertEqual(meta.num_key_value_heads, 3)

    def test_apply_rope_keeps_shape(self):
        cfg = AutoConfig.from_pretrained('EleutherAI/pythia-70m')
        meta = rope_metadata_from_config(cfg)
        x = np.ones((2, 3, meta.head_dim), dtype=np.float64)
        cos, sin = rope_cos_sin(meta, np.array([0, 1]))
        y = apply_rope(x, cos[:, None, :], sin[:, None, :], meta.rotary_dim)
        self.assertEqual(y.shape, x.shape)

    def test_relative_kernel_is_one_at_zero_gap(self):
        cfg = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B')
        meta = rope_metadata_from_config(cfg)
        kernel = relative_kernel(meta, np.array([0, 1, 2]))
        self.assertAlmostEqual(float(kernel[0]), 1.0)


if __name__ == '__main__':
    unittest.main()
