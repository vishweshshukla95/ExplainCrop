"""
ExplainCrop — Training loop.

Building this in pieces. Piece 1: the multi-task loss function.

Three heads, three losses, combined with the weights from config.py:
  - cause_loss:      cross-entropy over (healthy/deficiency/disease/pest),
                      trained on ALL crops including disease-only ones
  - deficiency_loss:  cross-entropy over NPK classes, but ONLY counted for
                       samples where cause == nutrient_deficiency (that's
                       what the ignore_index=-100 in dataset.py's npk_idx
                       is for — PyTorch's cross_entropy skips those rows)
  - severity_loss:     heteroscedastic regression loss from models.py,
                        only counted for samples with a real severity label
"""

import torch
import torch.nn.functional as F

from config import CFG
from models import heteroscedastic_loss


def compute_losses(model_out: dict, cause_idx: torch.Tensor,
                    npk_idx: torch.Tensor, severity_target: torch.Tensor,
                    severity_mask: torch.Tensor) -> dict:
    """
    model_out: dict from CropConditionedViT.forward()
    cause_idx: (B,) ground truth cause labels
    npk_idx:   (B,) ground truth NPK labels, -100 where not applicable
               (PyTorch's cross_entropy ignores -100 by default)
    severity_target: (B,) weak severity score in [0,1] from severity.py
    severity_mask:   (B,) 1.0 where a severity label exists, 0.0 otherwise
                      (can't use -100 trick for regression, so we mask manually)
    """
    cause_loss = F.cross_entropy(model_out["cause_logits"], cause_idx)

    deficiency_loss = F.cross_entropy(
        model_out["deficiency_logits"], npk_idx, ignore_index=-100)

    # Heteroscedastic severity loss, masked to only real labels.
    sev_mean = model_out["severity_mean"]
    sev_logvar = model_out["severity_logvar"]
    if severity_mask.sum() > 0:
        # Compute per-sample loss then average only over masked (labeled) rows.
        precision = torch.exp(-sev_logvar)
        per_sample = precision * (severity_target - sev_mean) ** 2 + sev_logvar
        severity_loss = (per_sample * severity_mask).sum() / severity_mask.sum()
    else:
        severity_loss = torch.tensor(0.0, device=sev_mean.device)

    w = CFG.loss_weights
    total = (w["cause"] * cause_loss
             + w["deficiency"] * deficiency_loss
             + w["severity"] * severity_loss)

    return {
        "total": total,
        "cause_loss": cause_loss,
        "deficiency_loss": deficiency_loss,
        "severity_loss": severity_loss,
    }


if __name__ == "__main__":
    # Sanity check with dummy tensors before wiring the real training loop.
    import torchvision.models as tvm
    _orig = tvm.vit_b_16
    tvm.vit_b_16 = lambda weights=None: _orig(weights=None)
    from models import CropConditionedViT

    model = CropConditionedViT()
    imgs = torch.randn(4, 3, 224, 224)
    crops = torch.tensor([0, 1, 2, 3])
    out = model(imgs, crops)

    cause_idx = torch.tensor([1, 2, 0, 1])
    npk_idx = torch.tensor([0, -100, -100, 2])       # crop 1,2 have no NPK label
    severity_target = torch.tensor([0.3, 0.0, 0.0, 0.6])
    severity_mask = torch.tensor([1.0, 0.0, 0.0, 1.0])

    losses = compute_losses(out, cause_idx, npk_idx, severity_target, severity_mask)
    for k, v in losses.items():
        print(k, float(v.detach()))
    print("Loss computation OK")
