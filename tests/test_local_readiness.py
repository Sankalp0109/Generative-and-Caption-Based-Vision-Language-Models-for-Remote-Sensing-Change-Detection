"""Comprehensive local readiness tests for Phase 7 CodeAug RSICC.

All tests run on CPU without any dataset downloads or model downloads.
The CodeAugRSICCModel uses fallback ViT/LLM backends for offline verification.
"""

import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── Source Imports ────────────────────────────────────────────────────────────

from src.config import DataConfig, ModelConfig, RemoteCLIPConfig, TrainConfig
from src.models.codeaug.config import CodeAugConfig
from src.models.codeaug.lora import VisualLoRALayer, inject_visual_lora
from src.models.codeaug.pos_embed import bicubic_interpolate_pos_embed
from src.models.codeaug.qformer import QFormerTokenCompressor, QFormerLayer
from src.models.codeaug.model import CodeAugRSICCModel
from src.codeaug_dataset import (
    compute_gli,
    BiTemporalUnionGLIJitter,
    get_codeaug_transforms,
    get_balanced_sampler,
    export_cider_idf,
    load_cider_idf,
)
from src.dataset import (
    Vocabulary,
    CaptionCollate,
    build_image_transforms,
    build_remoteclip_transforms,
    extract_ordered_patches_from_pil,
)
from src.training import (
    train_epoch,
    validate,
    generate_caption,
    save_checkpoint,
    load_checkpoint,
    build_optimizer_and_scheduler,
)
from src.utils import set_seed, get_device, stack_image_pair, denormalize_image


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_batch(batch_size=2, img_size=252):
    """Create a synthetic batch mimicking CaptionCollate output."""
    before = torch.randn(batch_size, 3, img_size, img_size)
    after = torch.randn(batch_size, 3, img_size, img_size)
    return {
        "before_images": before,
        "after_images": after,
        "images": torch.stack([before, after], dim=1),
        "caption_tokens": torch.ones(batch_size, 10, dtype=torch.long),
        "captions": [
            "a new building was constructed near the road",
            "the scene shows no change between the two dates",
        ][:batch_size],
        "changeflags": torch.tensor([1, 0][:batch_size]),
        "filenames": [f"img_{i}.png" for i in range(batch_size)],
    }


class FakeDataLoader:
    """Minimal iterable that yields the same batch n times."""
    def __init__(self, batch, n=2):
        self.batch = batch
        self.n = n
        self.dataset = type("DS", (), {"__len__": lambda self_: n * batch["before_images"].size(0)})()

    def __iter__(self):
        for _ in range(self.n):
            yield self.batch

    def __len__(self):
        return self.n


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUPS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigs(unittest.TestCase):
    """Verify all configuration dataclasses."""

    def test_data_config_defaults(self):
        cfg = DataConfig()
        self.assertEqual(cfg.img_size, (256, 256))
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.min_word_freq, 2)

    def test_data_config_post_init(self):
        cfg = DataConfig()
        self.assertTrue(str(cfg.caption_json).endswith("LevirCCcaptions.json"))
        self.assertEqual(cfg.val_batch_size, 32)  # batch_size * 2

    def test_model_config(self):
        cfg = ModelConfig()
        self.assertEqual(cfg.encoder_dim, 512)
        self.assertEqual(cfg.num_heads, 4)

    def test_remoteclip_config(self):
        cfg = RemoteCLIPConfig()
        self.assertEqual(cfg.model_name, "ViT-B-32")
        self.assertEqual(cfg.image_size, (224, 224))

    def test_train_config(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.learning_rate, 1e-4)
        self.assertEqual(cfg.num_epochs, 15)

    def test_codeaug_config(self):
        cfg = CodeAugConfig()
        self.assertEqual(cfg.img_size, (252, 252))
        self.assertEqual(cfg.grid_size, (18, 18))
        self.assertEqual(cfg.num_queries, 64)
        self.assertEqual(cfg.lora_r, 16)
        self.assertEqual(cfg.lora_alpha, 32)
        self.assertEqual(cfg.fusion_dim, 1024)
        self.assertEqual(cfg.vision_backbone, "ViT-L-14")

    def test_codeaug_config_custom(self):
        cfg = CodeAugConfig(num_queries=32, lora_r=8)
        self.assertEqual(cfg.num_queries, 32)
        self.assertEqual(cfg.lora_r, 8)


class TestVocabulary(unittest.TestCase):
    """Verify Vocabulary encode/decode and persistence."""

    def setUp(self):
        self.vocab = Vocabulary(min_freq=1)
        self.vocab.build_vocab([
            "a building was constructed",
            "no change observed in the scene",
            "trees were removed and a road was built",
        ])

    def test_special_tokens(self):
        self.assertEqual(self.vocab.pad_idx, 0)
        self.assertEqual(self.vocab.start_idx, 1)
        self.assertEqual(self.vocab.end_idx, 2)
        self.assertEqual(self.vocab.unk_idx, 3)

    def test_vocab_size(self):
        # 4 special + unique words
        self.assertGreater(len(self.vocab.word2idx), 4)

    def test_encode_decode_roundtrip(self):
        caption = "a building was constructed"
        encoded = self.vocab.encode(caption)
        self.assertEqual(encoded[0].item(), self.vocab.start_idx)
        self.assertEqual(encoded[-1].item(), self.vocab.end_idx)
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, caption)

    def test_unknown_word(self):
        encoded = self.vocab.encode("xyzunknownword")
        # Should contain UNK token
        self.assertIn(self.vocab.unk_idx, encoded.tolist())

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_vocab.pkl"
            self.vocab.save(path)
            loaded = Vocabulary.load(path)
            self.assertEqual(loaded.word2idx, self.vocab.word2idx)
            self.assertEqual(loaded.min_freq, self.vocab.min_freq)


class TestDataPipeline(unittest.TestCase):
    """Verify transforms, patch extraction, and CaptionCollate."""

    def test_image_transforms(self):
        tfm = build_image_transforms(img_size=(256, 256))
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        out = tfm(img)
        self.assertEqual(out.shape, (3, 256, 256))

    def test_remoteclip_transforms(self):
        tfm = build_remoteclip_transforms(img_size=(224, 224))
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        out = tfm(img)
        self.assertEqual(out.shape, (3, 224, 224))

    def test_codeaug_transforms(self):
        tfm = get_codeaug_transforms(img_size=(252, 252))
        img = Image.fromarray(np.uint8(np.random.rand(400, 400, 3) * 255))
        out = tfm(img)
        self.assertEqual(out.shape, (3, 252, 252))

    def test_extract_ordered_patches(self):
        img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
        patches = extract_ordered_patches_from_pil(img, patch_size=256)
        # 512/256 = 2x2 grid = 4 patches
        self.assertEqual(patches.shape, (4, 3, 256, 256))

    def test_extract_patches_with_padding(self):
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        patches = extract_ordered_patches_from_pil(img, patch_size=256)
        # ceil(300/256)=2 -> 2x2=4 patches with padding
        self.assertEqual(patches.shape, (4, 3, 256, 256))

    def test_caption_collate_shapes(self):
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(["hello world", "test caption"])
        collate = CaptionCollate(vocab, device="cpu")

        items = [
            {
                "before_image": torch.randn(3, 256, 256),
                "after_image": torch.randn(3, 256, 256),
                "caption": "hello world",
                "changeflag": 1,
                "filename": "test.png",
            },
            {
                "before_image": torch.randn(3, 256, 256),
                "after_image": torch.randn(3, 256, 256),
                "caption": "test caption",
                "changeflag": 0,
                "filename": "test2.png",
            },
        ]

        batch = collate(items)
        self.assertEqual(batch["before_images"].shape[0], 2)
        self.assertEqual(batch["after_images"].shape[0], 2)
        self.assertEqual(batch["images"].shape[0], 2)
        self.assertEqual(len(batch["captions"]), 2)
        self.assertEqual(batch["caption_tokens"].shape[0], 2)


class TestCodeAugDataPipeline(unittest.TestCase):
    """Verify GLI, jitter, sampler, and CIDEr IDF utilities."""

    def test_gli_3d(self):
        img = torch.rand(3, 64, 64)
        gli = compute_gli(img)
        self.assertEqual(gli.shape, (64, 64))
        self.assertTrue(gli.min() >= -1.0)
        self.assertTrue(gli.max() <= 1.0)

    def test_gli_4d(self):
        imgs = torch.rand(4, 3, 32, 32)
        gli = compute_gli(imgs)
        self.assertEqual(gli.shape, (4, 32, 32))

    def test_gli_invalid_dim(self):
        with self.assertRaises(ValueError):
            compute_gli(torch.rand(3, 3, 3, 3, 3))

    def test_bitemporal_jitter_output_size(self):
        jitter = BiTemporalUnionGLIJitter(p=1.0)  # Force application
        imgA = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        imgB = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        outA, outB = jitter(imgA, imgB)
        self.assertEqual(outA.size, (100, 100))
        self.assertEqual(outB.size, (100, 100))

    def test_bitemporal_jitter_skip(self):
        jitter = BiTemporalUnionGLIJitter(p=0.0)  # Force skip
        imgA = Image.fromarray(np.uint8(np.ones((50, 50, 3)) * 128))
        imgB = Image.fromarray(np.uint8(np.ones((50, 50, 3)) * 128))
        outA, outB = jitter(imgA, imgB)
        # Should return unchanged
        np.testing.assert_array_equal(np.array(outA), np.array(imgA))

    def test_balanced_sampler(self):
        # Create a fake dataset with known change flags
        class FakeDS:
            def __init__(self):
                self.samples = [
                    {"changeflag": 1},
                    {"changeflag": 0},
                    {"changeflag": 1},
                    {"changeflag": 0},
                    {"changeflag": 0},
                ]
            def __len__(self):
                return len(self.samples)

        sampler = get_balanced_sampler(FakeDS(), changed_to_unchanged_ratio=2.0)
        self.assertEqual(sampler.num_samples, 5)
        self.assertTrue(sampler.replacement)

    def test_cider_idf_export_load(self):
        samples = [
            {"sentences": [{"raw": "a building was built"}, {"raw": "new construction visible"}]},
            {"sentences": [{"raw": "no change in the scene"}, {"raw": "same area observed"}]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_idf.pkl"
            idf = export_cider_idf(samples, save_path=path)
            self.assertIsInstance(idf, dict)
            self.assertGreater(len(idf), 0)

            loaded = load_cider_idf(load_path=path)
            self.assertEqual(idf, loaded)

    def test_cider_idf_missing_file(self):
        result = load_cider_idf(load_path="/nonexistent/path.pkl")
        self.assertIsNone(result)


class TestVisualLoRA(unittest.TestCase):
    """Verify LoRA wrapper and injection."""

    def test_lora_layer_forward(self):
        base = nn.Linear(512, 512)
        lora = VisualLoRALayer(base, r=16, alpha=32, dropout=0.0)
        x = torch.randn(4, 512)
        out = lora(x)
        self.assertEqual(out.shape, (4, 512))

    def test_lora_layer_gradient_flow(self):
        base = nn.Linear(256, 256)
        lora = VisualLoRALayer(base, r=8, alpha=16)
        x = torch.randn(2, 256, requires_grad=True)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        # LoRA params should have gradients
        self.assertIsNotNone(lora.lora_A.grad)
        self.assertIsNotNone(lora.lora_B.grad)
        # Original layer should NOT have gradients (frozen)
        self.assertIsNone(base.weight.grad)

    def test_lora_properties(self):
        base = nn.Linear(64, 128, bias=True)
        lora = VisualLoRALayer(base, r=4, alpha=8)
        self.assertIs(lora.weight, base.weight)
        self.assertIs(lora.bias, base.bias)
        self.assertEqual(lora.scaling, 8 / 4)

    def test_inject_visual_lora(self):
        # Create a simple model with c_fc and c_proj layers
        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_fc = nn.Linear(64, 256)
                self.c_proj = nn.Linear(256, 64)
                self.other = nn.Linear(64, 64)

        model = SimpleMLP()
        replaced = inject_visual_lora(model, r=4, alpha=8, target_modules=("c_fc", "c_proj"))
        self.assertEqual(len(replaced), 2)
        self.assertIsInstance(model.c_fc, VisualLoRALayer)
        self.assertIsInstance(model.c_proj, VisualLoRALayer)
        self.assertNotIsInstance(model.other, VisualLoRALayer)


class TestPositionalEmbedding(unittest.TestCase):
    """Verify bicubic positional embedding interpolation."""

    def test_interpolate_2d(self):
        # (1 + 16*16, 64) = (257, 64) -> (1 + 18*18, 64) = (325, 64)
        pos = nn.Parameter(torch.randn(257, 64))
        new_pos = bicubic_interpolate_pos_embed(pos, new_grid_size=(18, 18))
        self.assertEqual(new_pos.shape, (325, 64))

    def test_interpolate_3d(self):
        # (1, 257, 64) -> (1, 325, 64)
        pos = nn.Parameter(torch.randn(1, 257, 64))
        new_pos = bicubic_interpolate_pos_embed(pos, new_grid_size=(18, 18))
        self.assertEqual(new_pos.shape, (1, 325, 64))

    def test_no_op_when_same_size(self):
        pos = nn.Parameter(torch.randn(325, 64))
        new_pos = bicubic_interpolate_pos_embed(pos, new_grid_size=(18, 18))
        self.assertTrue(torch.equal(pos.data, new_pos.data))

    def test_non_square_error(self):
        # 100 spatial tokens is not a perfect square (10*10=100 is, but 99 is not)
        pos = nn.Parameter(torch.randn(100, 64))  # 99 spatial + 1 CLS
        with self.assertRaises(ValueError):
            bicubic_interpolate_pos_embed(pos, new_grid_size=(12, 12))


class TestQFormer(unittest.TestCase):
    """Verify Q-Former token compression."""

    def test_compression_shape(self):
        cfg = CodeAugConfig()
        qf = QFormerTokenCompressor(config=cfg, llm_hidden_size=1536)
        visual_tokens = torch.randn(2, 324, 1024)
        out = qf(visual_tokens)
        self.assertEqual(out.shape, (2, 64, 1536))

    def test_single_layer(self):
        layer = QFormerLayer(embed_dim=512, num_heads=4)
        queries = torch.randn(2, 32, 512)
        visual = torch.randn(2, 100, 512)
        out = layer(queries, visual)
        self.assertEqual(out.shape, (2, 32, 512))

    def test_variable_input_length(self):
        cfg = CodeAugConfig()
        qf = QFormerTokenCompressor(config=cfg, llm_hidden_size=1536)
        # Input with 200 tokens instead of 324
        visual_tokens = torch.randn(1, 200, 1024)
        out = qf(visual_tokens)
        self.assertEqual(out.shape, (1, 64, 1536))

    def test_qformer_gradient_flow(self):
        cfg = CodeAugConfig()
        qf = QFormerTokenCompressor(config=cfg, llm_hidden_size=1536)
        visual_tokens = torch.randn(1, 324, 1024)
        out = qf(visual_tokens)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(qf.query_tokens.grad)


class TestCodeAugModel(unittest.TestCase):
    """Verify full CodeAugRSICCModel with fallback backends."""

    @classmethod
    def setUpClass(cls):
        """Create model once for all tests (fallback mode)."""
        os.environ["RSICC_FORCE_CPU"] = "1"
        cls.config = CodeAugConfig()
        cls.model = CodeAugRSICCModel(config=cls.config)
        cls.model.eval()
        cls.device = torch.device("cpu")

    def test_model_init(self):
        self.assertIsNotNone(self.model.vision_encoder)
        self.assertIsNotNone(self.model.qformer)
        self.assertIsNotNone(self.model.llm)
        self.assertIsNotNone(self.model.tokenizer)

    def test_tokenizer(self):
        tok = self.model.tokenizer
        result = tok(["hello world", "test"], return_tensors="pt", padding=True)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertEqual(result["input_ids"].shape[0], 2)

    def test_extract_visual_prefix(self):
        before = torch.randn(2, 3, 252, 252)
        after = torch.randn(2, 3, 252, 252)
        with torch.no_grad():
            prefix = self.model.extract_visual_prefix(before, after)
        self.assertEqual(prefix.shape, (2, 64, self.model.llm_hidden_size))

    def test_forward_pass(self):
        B = 2
        before = torch.randn(B, 3, 252, 252)
        after = torch.randn(B, 3, 252, 252)
        input_ids = torch.ones(B, 16, dtype=torch.long)
        attention_mask = torch.ones(B, 16, dtype=torch.long)
        labels = torch.ones(B, 16, dtype=torch.long)

        outputs = self.model(before, after, input_ids, attention_mask, labels)
        self.assertIsNotNone(outputs.loss)
        self.assertIsNotNone(outputs.logits)
        self.assertFalse(torch.isnan(outputs.loss))

    def test_generate_caption(self):
        before = torch.randn(2, 3, 252, 252)
        after = torch.randn(2, 3, 252, 252)
        captions = self.model.generate_caption(before, after)
        self.assertEqual(len(captions), 2)
        self.assertIsInstance(captions[0], str)
        self.assertGreater(len(captions[0]), 0)

    def test_trainable_params_exist(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        total = list(self.model.parameters())
        self.assertGreater(len(trainable), 0)
        # Some params should be frozen (ViT backbone)
        self.assertLess(len(trainable), len(total))


class TestTraining(unittest.TestCase):
    """Verify the modified training loop with CodeAug model."""

    def setUp(self):
        os.environ["RSICC_FORCE_CPU"] = "1"
        self.config = CodeAugConfig()
        self.model = CodeAugRSICCModel(config=self.config)
        self.device = torch.device("cpu")

    def test_train_epoch(self):
        batch = make_synthetic_batch(batch_size=2, img_size=252)
        loader = FakeDataLoader(batch, n=2)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        avg_loss = train_epoch(
            self.model, loader, optimizer, self.device,
            grad_clip=1.0, log_every=1, max_caption_len=32,
        )
        self.assertIsInstance(avg_loss, float)
        self.assertGreater(avg_loss, 0.0)
        self.assertFalse(np.isnan(avg_loss))

    def test_validate(self):
        batch = make_synthetic_batch(batch_size=2, img_size=252)
        loader = FakeDataLoader(batch, n=2)

        val_loss = validate(
            self.model, loader, self.device, max_caption_len=32,
        )
        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0.0)
        self.assertFalse(np.isnan(val_loss))

    def test_generate_caption_standalone(self):
        before = torch.randn(3, 252, 252)
        after = torch.randn(3, 252, 252)
        caption = generate_caption(self.model, before, after, self.device)
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)

    def test_gradient_flow(self):
        """Verify gradients flow to trainable params and NOT to frozen params."""
        batch = make_synthetic_batch(batch_size=1, img_size=252)

        # Zero all gradients
        self.model.zero_grad()

        before = batch["before_images"].to(self.device)
        after = batch["after_images"].to(self.device)
        captions = batch["captions"]

        tokenized = self.model.tokenizer(
            captions, return_tensors="pt", padding=True, truncation=True, max_length=32,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        labels = input_ids.clone()
        labels[labels == self.model.tokenizer.pad_token_id] = -100

        outputs = self.model(before, after, input_ids, attention_mask, labels)
        outputs.loss.backward()

        # Q-Former query tokens should have gradients
        self.assertIsNotNone(self.model.qformer.query_tokens.grad)

        # Frozen ViT backbone params should NOT have gradients
        for param in self.model.vision_encoder.parameters():
            if not param.requires_grad:
                self.assertIsNone(param.grad)


class TestCheckpoints(unittest.TestCase):
    """Verify checkpoint save/load roundtrip."""

    def test_save_load_roundtrip(self):
        os.environ["RSICC_FORCE_CPU"] = "1"
        config = CodeAugConfig()
        model = CodeAugRSICCModel(config=config)
        device = torch.device("cpu")

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(["test caption"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                model, optimizer, epoch=5, loss=0.123,
                vocab=vocab, checkpoint_dir=Path(tmpdir),
                name="test_model",
            )
            self.assertTrue(path.exists())

            model2 = CodeAugRSICCModel(config=config)
            opt2 = torch.optim.AdamW(
                [p for p in model2.parameters() if p.requires_grad], lr=1e-4,
            )
            model2, opt2, epoch, loaded_vocab, loss = load_checkpoint(
                model2, opt2, path, device,
            )
            self.assertEqual(epoch, 5)
            self.assertAlmostEqual(loss, 0.123, places=3)


class TestMetrics(unittest.TestCase):
    """Verify metric computation functions."""

    def test_bleu_scores(self):
        from src.metrics import compute_bleu_scores
        refs = [["a building was built"], ["no change observed"]]
        hyps = ["a building was built", "no change observed"]
        scores = compute_bleu_scores(refs, hyps)
        self.assertIn("BLEU-1", scores)
        self.assertIn("BLEU-4", scores)
        # Perfect match should give high scores
        self.assertGreater(scores["BLEU-1"], 0.5)

    def test_meteor_scores(self):
        from src.metrics import compute_meteor_scores
        refs = [["a building was built"]]
        hyps = ["a building was built"]
        score = compute_meteor_scores(refs, hyps)
        self.assertGreater(score, 0.5)

    def test_rouge_l_scores(self):
        from src.metrics import compute_rouge_l_scores
        refs = [["a building was built near the road"]]
        hyps = ["a building was built near the road"]
        scores = compute_rouge_l_scores(refs, hyps)
        self.assertIn("ROUGE-L", scores)
        self.assertAlmostEqual(scores["ROUGE-L"], 1.0, places=3)

    def test_cider_scores(self):
        from src.metrics import compute_cider_scores
        refs = [
            ["a building was built"],
            ["the area shows no change"],
        ]
        hyps = ["a building was built", "the area shows no change"]
        score = compute_cider_scores(refs, hyps)
        self.assertGreater(score, 0.0)

    def test_evaluate_single_pair(self):
        from src.metrics import evaluate_single_pair_metrics
        metrics = evaluate_single_pair_metrics(
            "a new building appeared", "a new building appeared"
        )
        self.assertIn("BLEU-1", metrics)
        self.assertIn("METEOR", metrics)
        self.assertIn("ROUGE-L", metrics)
        self.assertIn("CIDEr", metrics)

    def test_collect_references_and_predictions(self):
        from src.metrics import collect_references_and_predictions

        os.environ["RSICC_FORCE_CPU"] = "1"
        model = CodeAugRSICCModel(config=CodeAugConfig())
        model.eval()

        batch = make_synthetic_batch(batch_size=2, img_size=252)
        loader = FakeDataLoader(batch, n=1)

        refs, preds, details = collect_references_and_predictions(
            model, loader, torch.device("cpu"), max_samples=2,
        )
        self.assertEqual(len(refs), 2)
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(details), 2)
        self.assertIsInstance(preds[0], str)


class TestUtils(unittest.TestCase):
    """Verify utility functions."""

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        self.assertTrue(torch.equal(a, b))

    def test_get_device_cpu(self):
        os.environ["RSICC_FORCE_CPU"] = "1"
        device = get_device()
        self.assertEqual(device.type, "cpu")

    def test_stack_image_pair(self):
        before = torch.randn(4, 3, 256, 256)
        after = torch.randn(4, 3, 256, 256)
        stacked = stack_image_pair(before, after)
        self.assertEqual(stacked.shape, (4, 2, 3, 256, 256))

    def test_denormalize_image(self):
        img = torch.randn(3, 64, 64)
        result = denormalize_image(img)
        self.assertEqual(result.shape, (64, 64, 3))
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)


class TestEndToEndIntegration(unittest.TestCase):
    """Full pipeline integration test: init → train → validate → generate."""

    def test_full_pipeline(self):
        os.environ["RSICC_FORCE_CPU"] = "1"
        set_seed(42)
        device = torch.device("cpu")

        # 1. Config & Model
        config = CodeAugConfig()
        model = CodeAugRSICCModel(config=config)
        model.to(device)

        # 2. Synthetic data
        batch = make_synthetic_batch(batch_size=2, img_size=252)
        loader = FakeDataLoader(batch, n=2)

        # 3. Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        # 4. Train one epoch
        train_loss = train_epoch(
            model, loader, optimizer, device,
            grad_clip=1.0, log_every=1, max_caption_len=32,
        )
        self.assertIsInstance(train_loss, float)
        self.assertFalse(np.isnan(train_loss))

        # 5. Validate
        val_loss = validate(model, loader, device, max_caption_len=32)
        self.assertIsInstance(val_loss, float)
        self.assertFalse(np.isnan(val_loss))

        # 6. Generate caption
        before = torch.randn(3, 252, 252)
        after = torch.randn(3, 252, 252)
        caption = generate_caption(model, before, after, device)
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 0)

        # 7. Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                model, optimizer, epoch=1, loss=train_loss,
                vocab=None, checkpoint_dir=Path(tmpdir), name="integration_test",
            )
            self.assertTrue(path.exists())

        print(f"\\n✅ Integration test passed: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, caption='{caption}'")


if __name__ == "__main__":
    unittest.main()
