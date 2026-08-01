"""Test suite for Phase 7 CodeAug RSICC architecture."""

import unittest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from src.models import (
    CodeAugConfig,
    CodeAugRSICCModel,
    QFormerTokenCompressor,
    VisualLoRALayer,
)
from src.codeaug_dataset import (
    BiTemporalUnionGLIJitter,
    compute_gli,
    get_codeaug_transforms,
)


class TestPhase7CodeAug(unittest.TestCase):
    def test_config(self):
        cfg = CodeAugConfig()
        self.assertEqual(cfg.img_size, (252, 252))
        self.assertEqual(cfg.num_queries, 64)
        self.assertEqual(cfg.lora_r, 16)
        self.assertEqual(cfg.grid_size, (18, 18))

    def test_qformer_compression_shape(self):
        cfg = CodeAugConfig()
        qf = QFormerTokenCompressor(config=cfg, llm_hidden_size=1536)
        dummy_in = torch.randn(2, 324, 1024)
        out = qf(dummy_in)
        self.assertEqual(out.shape, (2, 64, 1536))

    def test_visual_lora_wrapper(self):
        base_lin = nn.Linear(512, 512)
        lora = VisualLoRALayer(base_lin, r=16, alpha=32)
        x = torch.randn(2, 512)
        out = lora(x)
        self.assertEqual(out.shape, (2, 512))

    def test_gli_jitter(self):
        jitter = BiTemporalUnionGLIJitter()
        imgA = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        imgB = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        outA, outB = jitter(imgA, imgB)
        self.assertEqual(outA.size, (100, 100))
        self.assertEqual(outB.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
