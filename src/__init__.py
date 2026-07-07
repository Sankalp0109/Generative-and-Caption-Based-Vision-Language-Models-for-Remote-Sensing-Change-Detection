"""
RSICC shared module library.

All three researchers should import from here so dataset format,
vocabulary, and training utilities stay identical across ablation runs.
"""

from .config import DataConfig, ModelConfig, RemoteCLIPConfig, TrainConfig
from .dataset import (
    CaptionCollate,
    LEVIRCCDataset,
    Vocabulary,
    build_image_transforms,
    build_remoteclip_transforms,
    build_vocabulary_from_annotations,
    get_levircc_loaders,
    load_levircc_annotations,
    split_samples_by_split,
)
from .models import (
    ChangeCaptioningModel,
    RSICCformerBaseline,
    SimpleDecoder,
    SimpleEncoder,
)
from .models.variants import RemoteCLIPDifferenceModel, RemoteCLIPEncoder
from .training import (
    build_criterion,
    build_optimizer_and_scheduler,
    generate_caption,
    load_checkpoint,
    prepare_teacher_forcing_inputs,
    save_checkpoint,
    train_epoch,
    validate,
    visualize_predictions,
    get_checkpoint_epoch
)
from .metrics import (
    SentenceEmbeddingScorer,
    collect_references_and_predictions,
    compute_bleu_scores,
    compute_cider_scores,
    compute_meteor_scores,
    compute_rouge_l_scores,
    evaluate_caption_metrics,
    evaluate_full_test_with_per_sample_metrics,
    evaluate_model_on_loader,
    evaluate_single_pair_metrics,
    generate_caption_greedy,
    save_phase1_results,
)
from .utils import (
    cuda_device_is_compatible,
    denormalize_image,
    get_device,
    set_seed,
    setup_project_path,
    stack_image_pair,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "RemoteCLIPConfig",
    "TrainConfig",
    "Vocabulary",
    "LEVIRCCDataset",
    "CaptionCollate",
    "get_levircc_loaders",
    "load_levircc_annotations",
    "split_samples_by_split",
    "build_vocabulary_from_annotations",
    "build_image_transforms",
    "build_remoteclip_transforms",
    "ChangeCaptioningModel",
    "SimpleEncoder",
    "SimpleDecoder",
    "RSICCformerBaseline",
    "RemoteCLIPEncoder",
    "RemoteCLIPDifferenceModel",
    "train_epoch",
    "validate",
    "save_checkpoint",
    "load_checkpoint",
    "generate_caption",
    "visualize_predictions",
    "prepare_teacher_forcing_inputs",
    "build_criterion",
    "build_optimizer_and_scheduler",
    "SentenceEmbeddingScorer",
    "collect_references_and_predictions",
    "compute_bleu_scores",
    "compute_cider_scores",
    "compute_meteor_scores",
    "compute_rouge_l_scores",
    "evaluate_caption_metrics",
    "evaluate_model_on_loader",
    "evaluate_full_test_with_per_sample_metrics",
    "evaluate_single_pair_metrics",
    "save_phase1_results",
    "generate_caption_greedy",
    "setup_project_path",
    "set_seed",
    "get_device",
    "cuda_device_is_compatible",
    "stack_image_pair",
    "denormalize_image",
    "get_checkpoint_epoch"
]
