"""Captioning metrics and semantic similarity evaluation for RSICC.

This module provides lightweight, reusable evaluation utilities for all
researchers in the ablation study.

Supported metrics:
- BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
- METEOR
- ROUGE-L
- CIDEr-style score (lightweight TF-IDF cosine approximation)
- Sentence embedding similarity (SentenceTransformer / Transformers / TF-IDF fallback)

The goal is to keep the evaluation protocol shared and reproducible across
all model variants.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer


SPECIAL_TOKENS = {"<PAD>", "<START>", "<END>", "<UNK>"}


@dataclass
class MetricBundle:
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    meteor: float
    rouge_l: float
    cider: float
    semantic_similarity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "BLEU-1": self.bleu1,
            "BLEU-2": self.bleu2,
            "BLEU-3": self.bleu3,
            "BLEU-4": self.bleu4,
            "METEOR": self.meteor,
            "ROUGE-L": self.rouge_l,
            "CIDEr": self.cider,
            "Semantic-Similarity": self.semantic_similarity,
        }


def normalize_caption(text: str) -> str:
    """Basic normalization shared across metrics."""
    return " ".join(text.lower().strip().split())


def normalize_references(
    references: Sequence[Sequence[str] | str],
) -> List[List[str]]:
    """Normalize refs into list of reference lists."""
    normalized = []
    for ref in references:
        if isinstance(ref, str):
            normalized.append([normalize_caption(ref)])
        else:
            normalized.append([normalize_caption(r) for r in ref])
    return normalized


def normalize_hypotheses(hypotheses: Sequence[str]) -> List[str]:
    return [normalize_caption(h) for h in hypotheses]


def tokenize(text: str) -> List[str]:
    return normalize_caption(text).split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _lcs_length(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_bleu_scores(
    references: Sequence[Sequence[str] | str],
    hypotheses: Sequence[str],
) -> Dict[str, float]:
    """Compute BLEU-1..4 using NLTK corpus BLEU."""
    refs = normalize_references(references)
    hyps = normalize_hypotheses(hypotheses)

    smooth = SmoothingFunction().method4
    ref_tokens = [[tokenize(r) for r in ref_list] for ref_list in refs]
    hyp_tokens = [tokenize(h) for h in hyps]

    bleu1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": float(bleu1),
        "BLEU-2": float(bleu2),
        "BLEU-3": float(bleu3),
        "BLEU-4": float(bleu4),
    }


def compute_meteor_scores(
    references: Sequence[Sequence[str] | str],
    hypotheses: Sequence[str],
) -> float:
    """Compute average METEOR score."""
    refs = normalize_references(references)
    hyps = normalize_hypotheses(hypotheses)

    scores = []
    for ref_list, hyp in zip(refs, hyps):
        score = meteor_score([tokenize(ref) for ref in ref_list],tokenize(hyp))
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def _fallback_overlap_f1(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    ref_counts = Counter(ref)
    hyp_counts = Counter(hyp)
    overlap = sum(min(ref_counts[t], hyp_counts[t]) for t in ref_counts)
    precision = overlap / max(len(hyp), 1)
    recall = overlap / max(len(ref), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l_scores(
    references: Sequence[Sequence[str] | str],
    hypotheses: Sequence[str],
) -> Dict[str, float]:
    """Compute average ROUGE-L precision/recall/F1 against the best reference."""
    refs = normalize_references(references)
    hyps = normalize_hypotheses(hypotheses)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for ref_list, hyp in zip(refs, hyps):
        hyp_tokens = tokenize(hyp)
        best = (0.0, 0.0, 0.0)
        for ref in ref_list:
            ref_tokens = tokenize(ref)
            lcs = _lcs_length(ref_tokens, hyp_tokens)
            prec = lcs / max(len(hyp_tokens), 1)
            rec = lcs / max(len(ref_tokens), 1)
            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = (2 * prec * rec) / (prec + rec)
            if f1 > best[2]:
                best = (prec, rec, f1)
        precisions.append(best[0])
        recalls.append(best[1])
        f1s.append(best[2])

    return {
        "ROUGE-L_P": float(np.mean(precisions)) if precisions else 0.0,
        "ROUGE-L_R": float(np.mean(recalls)) if recalls else 0.0,
        "ROUGE-L": float(np.mean(f1s)) if f1s else 0.0,
    }


def _build_idf_weights(documents: List[List[str]], n: int) -> Dict[Tuple[str, ...], float]:
    """Compute IDF for n-grams across reference corpus."""
    df = Counter()
    total_docs = len(documents)
    for tokens in documents:
        seen = set(_ngram_counts(tokens, n))
        for gram in seen:
            df[gram] += 1

    idf = {}
    for gram, freq in df.items():
        idf[gram] = np.log((total_docs + 1) / (freq + 1))
    return idf


def _tfidf_vector(tokens: List[str], n: int, idf: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
    counts = _ngram_counts(tokens, n)
    total = sum(counts.values()) or 1
    vec = {}
    for gram, c in counts.items():
        if gram in idf:
            tf = c / total
            vec[gram] = tf * idf[gram]
    return vec


def _cosine_sparse(v1: Dict[Tuple[str, ...], float], v2: Dict[Tuple[str, ...], float]) -> float:
    if not v1 or not v2:
        return 0.0
    keys = set(v1) & set(v2)
    num = sum(v1[k] * v2[k] for k in keys)
    den1 = sum(v * v for v in v1.values())
    den2 = sum(v * v for v in v2.values())
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (np.sqrt(den1) * np.sqrt(den2))


def compute_cider_scores(
    references: Sequence[Sequence[str] | str],
    hypotheses: Sequence[str],
    max_n: int = 4,
) -> float:
    """Lightweight CIDEr-style score using TF-IDF cosine over n-grams.

    This is a project-friendly approximation of CIDEr when the full COCO
    evaluation package is not available.
    """
    refs = normalize_references(references)
    hyps = normalize_hypotheses(hypotheses)

    ref_docs = [tokenize(ref_list[0]) for ref_list in refs]
    idf_by_n = {n: _build_idf_weights(ref_docs, n) for n in range(1, max_n + 1)}

    scores = []
    for ref_list, hyp in zip(refs, hyps):
        hyp_tokens = tokenize(hyp)
        per_n_scores = []
        for n in range(1, max_n + 1):
            hyp_vec = _tfidf_vector(hyp_tokens, n, idf_by_n[n])
            ref_vecs = [_tfidf_vector(tokenize(ref), n, idf_by_n[n]) for ref in ref_list]
            if ref_vecs:
                ref_vec = defaultdict(float)
                for vec in ref_vecs:
                    for k, v in vec.items():
                        ref_vec[k] += v / len(ref_vecs)
                per_n_scores.append(_cosine_sparse(hyp_vec, dict(ref_vec)))
            else:
                per_n_scores.append(0.0)
        scores.append(10.0 * float(np.mean(per_n_scores)))

    return float(np.mean(scores)) if scores else 0.0


class SentenceEmbeddingScorer:
    """Semantic similarity scorer based on sentence embeddings.

    Preferred backend order:
        1. sentence-transformers (all-MiniLM-L6-v2)
        2. transformers AutoModel + mean pooling
        3. sklearn TF-IDF cosine fallback
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self._model = None
        self._tokenizer = None
        self._vectorizer = None

    def _load_backend(self):
        if self.backend is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.backend = "sentence-transformers"

    def score(self, references: Sequence[Sequence[str] | str], hypotheses: Sequence[str]) -> float:
        """Return mean cosine similarity between predictions and references."""
        return float(np.mean(self.score_pairs(references, hypotheses))) if hypotheses else 0.0

    def score_pairs(
        self,
        references: Sequence[Sequence[str] | str],
        hypotheses: Sequence[str],
    ) -> List[float]:
        """Return cosine similarity for each reference/prediction pair."""
        refs = normalize_references(references)
        hyps = normalize_hypotheses(hypotheses)
        if not hyps:
            return []

        self._load_backend()
        ref_texts = [ref_list[0] for ref_list in refs]
        hyp_emb = self._model.encode(hyps, convert_to_numpy=True, normalize_embeddings=True)
        ref_emb = self._model.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.sum(hyp_emb * ref_emb, axis=1).astype(float).tolist()


def evaluate_caption_metrics(
    references: Sequence[Sequence[str] | str],
    hypotheses: Sequence[str],
    semantic_model: Optional[SentenceEmbeddingScorer] = None,
) -> Dict[str, float]:
    """Compute the full metric bundle for one set of references/hypotheses."""
    bleu = compute_bleu_scores(references, hypotheses)
    meteor = compute_meteor_scores(references, hypotheses)
    rouge = compute_rouge_l_scores(references, hypotheses)
    cider = compute_cider_scores(references, hypotheses)
    semantic_model = semantic_model or SentenceEmbeddingScorer()
    semantic_similarity = semantic_model.score(references, hypotheses)

    bundle = MetricBundle(
        bleu1=bleu["BLEU-1"],
        bleu2=bleu["BLEU-2"],
        bleu3=bleu["BLEU-3"],
        bleu4=bleu["BLEU-4"],
        meteor=meteor,
        rouge_l=rouge["ROUGE-L"],
        cider=cider,
        semantic_similarity=semantic_similarity,
    )
    return bundle.as_dict()


@torch.no_grad()
def generate_caption_greedy(model, before_image, after_image, vocab, device, max_len: int = 100) -> str:
    """Generate a caption from a single before/after pair using greedy decoding."""
    model.eval()
    temporal_dim = 1 if before_image.ndim == 4 else 0
    images = torch.stack([before_image, after_image], dim=temporal_dim).unsqueeze(0).to(device)
    caption_tokens = [vocab.start_idx]

    for _ in range(max_len):
        cap_tensor = torch.tensor([caption_tokens], dtype=torch.long, device=device)
        logits = model(images, cap_tensor)
        next_token = logits[0, -1, :].argmax(-1).item()
        caption_tokens.append(next_token)
        if next_token == vocab.end_idx:
            break

    return vocab.decode(caption_tokens)


@torch.no_grad()
def collect_references_and_predictions(
    model,
    data_loader,
    vocab,
    device,
    max_samples: Optional[int] = None,
) -> Tuple[List[List[str]], List[str], List[Dict]]:
    """Run caption generation over a loader and collect refs/preds/details."""
    model.eval()
    references: List[List[str]] = []
    predictions: List[str] = []
    details: List[Dict] = []
    seen = 0

    for batch in data_loader:
        before_images = batch["before_images"]
        after_images = batch["after_images"]
        captions = batch["captions"]
        filenames = batch.get("filenames", [None] * len(captions))
        changeflags = batch.get("changeflags", [None] * len(captions))

        for idx in range(before_images.size(0)):
            ref = captions[idx]
            pred = generate_caption_greedy(
                model,
                before_images[idx],
                after_images[idx],
                vocab,
                device,
            )
            references.append([ref])
            predictions.append(pred)
            details.append(
                {
                    "filename": filenames[idx],
                    "changeflag": int(changeflags[idx]) if changeflags[idx] is not None else None,
                    "reference": ref,
                    "prediction": pred,
                }
            )
            seen += 1
            if max_samples is not None and seen >= max_samples:
                return references, predictions, details

    return references, predictions, details


def evaluate_model_on_loader(
    model,
    data_loader,
    vocab,
    device,
    max_samples: Optional[int] = None,
    semantic_model: Optional[SentenceEmbeddingScorer] = None,
) -> Dict[str, float]:
    """Collect model predictions and compute all caption metrics."""
    references, predictions, details = collect_references_and_predictions(
        model=model,
        data_loader=data_loader,
        vocab=vocab,
        device=device,
        max_samples=max_samples,
    )
    metrics = evaluate_caption_metrics(
        references=references,
        hypotheses=predictions,
        semantic_model=semantic_model,
    )
    metrics["num_samples"] = len(predictions)
    return metrics, details


def evaluate_single_pair_metrics(reference: str, prediction: str) -> Dict[str, float]:
    """Compute caption metrics for one reference/prediction pair."""
    bleu = compute_bleu_scores([reference], [prediction])
    meteor = compute_meteor_scores([reference], [prediction])
    rouge = compute_rouge_l_scores([reference], [prediction])
    cider = compute_cider_scores([reference], [prediction])
    return {
        **bleu,
        "METEOR": meteor,
        "ROUGE-L": rouge["ROUGE-L"],
        "CIDEr": cider,
    }


def attach_per_sample_metrics(
    details: List[Dict],
    references: Sequence[Sequence[str]],
    predictions: Sequence[str],
    semantic_model: Optional[SentenceEmbeddingScorer] = None,
) -> List[Dict]:
    """Attach per-sample metrics to prediction detail records."""
    semantic_scores: Optional[List[float]] = None
    if semantic_model is not None:
        semantic_scores = semantic_model.score_pairs(references, predictions)

    enriched: List[Dict] = []
    for idx, (detail, ref_list, prediction) in enumerate(zip(details, references, predictions)):
        metrics = evaluate_single_pair_metrics(ref_list[0], prediction)
        if semantic_scores is not None:
            metrics["Semantic-Similarity"] = semantic_scores[idx]
        enriched.append({**detail, "metrics": metrics})
    return enriched


def evaluate_full_test_with_per_sample_metrics(
    model,
    data_loader,
    vocab,
    device,
    semantic_model: Optional[SentenceEmbeddingScorer] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Run inference on a loader and return aggregate + per-sample metrics."""
    references, predictions, details = collect_references_and_predictions(
        model=model,
        data_loader=data_loader,
        vocab=vocab,
        device=device,
        max_samples=max_samples,
    )
    aggregate_metrics = evaluate_caption_metrics(
        references=references,
        hypotheses=predictions,
        semantic_model=semantic_model,
    )
    aggregate_metrics["num_samples"] = len(predictions)

    samples = attach_per_sample_metrics(
        details=details,
        references=references,
        predictions=predictions,
        semantic_model=semantic_model,
    )
    return aggregate_metrics, samples


def build_phase1_results_payload(
    aggregate_metrics: Dict[str, float],
    samples: List[Dict],
    metadata: Optional[Dict] = None,
    preview_count: int = 40,
) -> Dict:
    """Build a JSON-serializable Phase 1 results object."""
    payload: Dict = {
        "metadata": metadata or {},
        "full_test_metrics": aggregate_metrics,
        "full_test_count": len(samples),
        "full_test_samples": samples,
    }

    if preview_count > 0 and samples:
        preview = samples[:preview_count]
        preview_refs = [[item["reference"]] for item in preview]
        preview_preds = [item["prediction"] for item in preview]
        payload["sample_40_metrics"] = evaluate_caption_metrics(
            references=preview_refs,
            hypotheses=preview_preds,
        )
        payload["sample_40_metrics"]["num_samples"] = len(preview)
        payload["sample_40_details"] = [
            {
                "filename": item["filename"],
                "changeflag": item["changeflag"],
                "reference": item["reference"],
                "prediction": item["prediction"],
            }
            for item in preview
        ]
    return payload


def save_phase1_results(
    output_path: Path,
    aggregate_metrics: Dict[str, float],
    samples: List[Dict],
    metadata: Optional[Dict] = None,
    preview_count: int = 40,
) -> Path:
    """Save Phase 1 evaluation output with per-sample metrics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_phase1_results_payload(
        aggregate_metrics=aggregate_metrics,
        samples=samples,
        metadata=metadata,
        preview_count=preview_count,
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)
        file.write("\n")

    print(f"Saved Phase 1 results to {output_path.resolve()}")
    print(f"  Full test samples: {len(samples)}")
    return output_path
