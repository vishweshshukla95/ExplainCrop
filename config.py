"""
ExplainCrop — Central configuration.

Single source of truth for the class taxonomy, crop list, model
hyperparameters, and paths. Every other module imports from here so the
taxonomy never drifts between dataset.py / models.py / recommendation.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Crop / class taxonomy
# ---------------------------------------------------------------------------
# Crops with usable public NPK-deficiency data (your primary training set).
CROPS_PRIMARY = ["rice", "maize", "tomato", "wheat"]

# Cucurbits from the EarlyNSD dataset — real deficiency labels (N, K only;
# no phosphorus class exists in this dataset), early-stage/subtle symptoms.
CROPS_EARLYNSD = ["ashgourd", "bittergourd", "snakegourd"]

# Crops used ONLY for the disentanglement head (disease/pest images, no
# deficiency labels needed) — widens robustness without needing deficiency
# annotations for every crop.
CROPS_DISEASE_ONLY = ["cotton"]

# Crops with rich micronutrient data — used to pretrain the deficiency head
# on a larger label space before fine-tuning on the NPK crops.
CROPS_MICRONUTRIENT = ["banana", "coffee"]

# Crop with NO deficiency data at all — the few-shot transfer target.
CROP_FEWSHOT_TARGET = "cotton"

ALL_CROPS = CROPS_PRIMARY + CROPS_EARLYNSD + CROPS_DISEASE_ONLY + CROPS_MICRONUTRIENT
CROP_TO_IDX = {c: i for i, c in enumerate(ALL_CROPS)}

# "Cause" head — the disease-vs-deficiency disentanglement classes.
CAUSE_CLASSES = ["healthy", "nutrient_deficiency", "disease", "pest"]

# NPK deficiency classes shared across the primary crops. Zinc was added
# because the maize dataset (ZNAB) has a real, well-populated zinc-deficiency
# class, and zinc already exists in MICRONUTRIENT_CLASSES below — this lets
# maize contribute to the micronutrient objective too, not just macro-NPK.
NPK_CLASSES = ["healthy", "nitrogen", "phosphorus", "potassium", "zinc"]

# Extended micronutrient classes (banana/coffee pretraining only).
MICRONUTRIENT_CLASSES = [
    "healthy", "nitrogen", "phosphorus", "potassium",
    "boron", "calcium", "iron", "magnesium", "manganese", "sulfur", "zinc",
]

SEVERITY_LEVELS = ["low", "moderate", "high"]


@dataclass
class Config:
    # --- data ---
    data_root: str = "datasets"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # --- model ---
    backbone: str = "vit_b_16"          # alt: "swin_t"
    embed_dim: int = 768
    crop_token_dim: int = 64            # learned crop-conditioning embedding
    num_causes: int = len(CAUSE_CLASSES)
    num_npk_classes: int = len(NPK_CLASSES)
    num_severity_levels: int = len(SEVERITY_LEVELS)

    # --- training ---
    epochs: int = 50
    lr: float = 3e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "cause": 1.0,
        "deficiency": 1.0,
        "severity": 0.5,
    })

    # --- few-shot (cotton transfer) ---
    fewshot_k_shot: int = 5             # labeled examples per class
    fewshot_episodes: int = 200

    # --- explainability ---
    # CropConditionedViT (models.py) unpacks vit.encoder into three
    # separate attributes (encoder_dropout, encoder_layers, encoder_ln)
    # rather than keeping vit.encoder as one module -- so the hook path
    # is "encoder_layers.encoder_layer_11", not torchvision's original
    # "encoder.layers.encoder_layer_11". Confirmed by inspection, not guessed.
    gradcam_target_layer: str = "encoder_layers.encoder_layer_11"

    # --- paths ---
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"


CFG = Config()