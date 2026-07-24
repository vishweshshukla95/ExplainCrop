"""
ExplainCrop — Model architecture.

This is where the actual novelty lives. Three design choices separate this
from a standard "ViT + classification head" baseline:

1. CROP-CONDITIONED ATTENTION
   A learned crop embedding is injected as an extra token into the ViT
   sequence (like a CLS token, but crop-specific). Self-attention lets every
   patch token attend to this crop token, so the same nitrogen-deficiency
   visual features get interpreted differently depending on which crop
   they're coming from — this is what makes "one model, many crops" work
   instead of collapsing all crops into one blurry decision boundary.

2. TWO-STAGE DISENTANGLEMENT
   Stage 1 (cause head) asks "is this healthy / a deficiency / a disease /
   a pest problem?" using ALL crops, including disease-only crops like
   cotton that have zero deficiency labels. Stage 2 (deficiency head) only
   fires its loss when cause == nutrient_deficiency, and is trained only on
   crops that have deficiency labels. Gradients from disease-only crops
   still shape the shared backbone via the cause head — that's how cotton
   contributes to the model despite having no deficiency data.

3. SEVERITY AS A DISTRIBUTION, NOT A POINT ESTIMATE
   The severity head outputs a mean and log-variance (heteroscedastic
   regression) instead of a single number, so recommendation.py can scale
   down fertilizer dosage confidence when the model is unsure — this is the
   link between explainability/uncertainty and the recommendation engine.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from config import CFG, ALL_CROPS


class CropConditionedViT(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

        # Backbone: torchvision ViT-B/16, pretrained on ImageNet.
        vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self._vit_ref = vit  # keep for _process_input() reuse

        # We do NOT call vit.encoder(x) directly. torchvision's
        # Encoder.forward does `input + self.pos_embedding` with a FIXED
        # (1, 197, 768) tensor -- 197 = 196 patches + 1 CLS, sized for the
        # pretrained checkpoint. Our sequence is 198 tokens (patches + CLS
        # + crop token), so that add would break on a shape mismatch.
        # Confirmed by inspecting torchvision.models.vision_transformer's
        # source directly rather than assuming.
        #
        # Fix: build our own (1, 198, 768) pos_embedding, copying the
        # pretrained weights into the CLS + patch positions and adding one
        # new learnable position for the crop token (inserted at index 1,
        # right after CLS). Then reuse the pretrained dropout/layers/ln
        # directly, since those don't depend on sequence length.
        pretrained_pos = vit.encoder.pos_embedding.detach().clone()  # (1, 197, 768)
        cls_pos = pretrained_pos[:, :1, :]                            # (1, 1, 768)
        patch_pos = pretrained_pos[:, 1:, :]                          # (1, 196, 768)
        crop_pos = torch.zeros_like(cls_pos)                          # new, learnable

        self.pos_embedding = nn.Parameter(
            torch.cat([cls_pos, crop_pos, patch_pos], dim=1))         # (1, 198, 768)

        self.encoder_dropout = vit.encoder.dropout
        self.encoder_layers = vit.encoder.layers
        self.encoder_ln = vit.encoder.ln

        # Learned crop embedding -> projected to a token in the same space
        # as patch embeddings. This is the "crop-conditioning" mechanism.
        self.crop_embed = nn.Embedding(len(ALL_CROPS), cfg.crop_token_dim)
        self.crop_proj = nn.Linear(cfg.crop_token_dim, cfg.embed_dim)

        d = cfg.embed_dim

        # Stage 1: cause classifier (healthy / deficiency / disease / pest)
        self.cause_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, cfg.num_causes),
        )

        # Stage 2: NPK deficiency-type classifier (only meaningful when
        # cause == nutrient_deficiency; loss is masked accordingly).
        self.deficiency_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
            nn.Linear(256, cfg.num_npk_classes),
        )

        # Severity: heteroscedastic regression -> (mean, log_var)
        self.severity_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, images: torch.Tensor, crop_idx: torch.Tensor):
        """
        images:   (B, 3, H, W)
        crop_idx: (B,) long tensor of crop indices (see config.CROP_TO_IDX)
        """
        vit = self._vit_ref
        x = vit._process_input(images)              # (B, N, D) patch tokens
        b = x.shape[0]

        cls = self.cls_token.expand(b, -1, -1)       # (B, 1, D)
        crop_tok = self.crop_proj(self.crop_embed(crop_idx)).unsqueeze(1)  # (B, 1, D)

        x = torch.cat([cls, crop_tok, x], dim=1)      # (B, 198, D): [CLS, crop, patches...]
        x = x + self.pos_embedding                     # our custom (1, 198, D) embedding
        x = self.encoder_ln(self.encoder_layers(self.encoder_dropout(x)))
        pooled = x[:, 0]                                # CLS output

        cause_logits = self.cause_head(pooled)
        deficiency_logits = self.deficiency_head(pooled)
        severity_mean_logvar = self.severity_head(pooled)

        return {
            "cause_logits": cause_logits,
            "deficiency_logits": deficiency_logits,
            "severity_mean": severity_mean_logvar[:, 0],
            "severity_logvar": severity_mean_logvar[:, 1],
            "features": pooled,   # exposed for Grad-CAM / few-shot prototypes
        }


def heteroscedastic_loss(pred_mean, pred_logvar, target):
    """Negative log-likelihood under a Gaussian with predicted variance.

    Forces the model to predict LOW variance only when it's actually
    accurate -- this is what gives recommendation.py a genuine confidence
    signal instead of a hand-waved one.
    """
    precision = torch.exp(-pred_logvar)
    return (precision * (target - pred_mean) ** 2 + pred_logvar).mean()


if __name__ == "__main__":
    model = CropConditionedViT()
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_crop = torch.tensor([0, 1])
    out = model(dummy_img, dummy_crop)
    for k, v in out.items():
        print(k, v.shape)
