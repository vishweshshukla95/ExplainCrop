"""
ExplainCrop — Explainability (Grad-CAM).

Building this in pieces. Piece 1: hook registration.

WHY THIS IS NEEDED:
Grad-CAM needs two things from inside the model, at inference time:
  1. The activation (output) of a chosen internal layer
  2. The gradient of the prediction w.r.t. that same activation

PyTorch doesn't expose these by default — you have to explicitly "hook"
onto the layer and capture them as the data flows through. That's all this
piece does: no CAM math yet, just capturing the two tensors we'll need.

We target the last ViT encoder block (config.CFG.gradcam_target_layer) —
this is standard practice for transformers, since the last block's patch
tokens still carry the most spatial information before pooling.
"""

import torch

from config import CFG


class GradCAMHooks:
    """Registers forward + backward hooks on one layer and stores the results."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = CFG.gradcam_target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = self._get_layer(model, target_layer_name)

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _get_layer(self, model: torch.nn.Module, layer_name: str):
        """Resolves a dotted layer path like 'encoder.layers.encoder_layer_11'
        into the actual submodule object."""
        module = model
        for attr in layer_name.split("."):
            module = getattr(module, attr)
        return module

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()


# ---------------------------------------------------------------------------
# Piece 2: turning captured activations/gradients into a CAM heatmap.
# ---------------------------------------------------------------------------
import numpy as np
import cv2


class GradCAM:
    """Computes a Grad-CAM heatmap for one image given a target output.

    ViT-specific detail: activations are (B, N_tokens, D), not the
    (B, C, H, W) feature maps CNN Grad-CAM expects. We drop the CLS and
    crop tokens (positions 0 and 1), keep only the 196 patch tokens, and
    reshape them back into a 14x14 spatial grid (14*14=196, matching
    ViT-B/16's 16x16 patch stride on a 224x224 image) before computing the
    weighted sum — this reshape is the actual "ViT Grad-CAM" adaptation.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = CFG.gradcam_target_layer):
        self.model = model
        self.hooks = GradCAMHooks(model, target_layer_name)

    def __call__(self, image: torch.Tensor, crop_idx: torch.Tensor,
                 target: str = "cause", target_class: int = None):
        """
        image:    (1, 3, 224, 224)
        crop_idx: (1,)
        target:   "cause" or "deficiency" — which head to explain
        target_class: index to explain; if None, uses the predicted class
        """
        self.model.zero_grad()
        out = self.model(image, crop_idx)

        logits = out["cause_logits"] if target == "cause" else out["deficiency_logits"]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        acts = self.hooks.activations       # (1, 198, 768)
        grads = self.hooks.gradients         # (1, 198, 768)

        # Drop CLS (index 0) and crop token (index 1); keep 196 patch tokens.
        patch_acts = acts[0, 2:, :]          # (196, 768)
        patch_grads = grads[0, 2:, :]        # (196, 768)

        # Global-average-pool gradients over the embedding dim -> per-token
        # importance weight, then weight the activations and sum -- this is
        # the direct token-sequence analogue of standard Grad-CAM's
        # channel-wise pooling over spatial feature maps.
        weights = patch_grads.mean(dim=1)              # (196,)
        cam = (weights.unsqueeze(1) * patch_acts).sum(dim=1)  # (196,)
        cam = torch.relu(cam)

        grid = cam.reshape(14, 14).detach().numpy()
        grid = grid / (grid.max() + 1e-8)               # normalize to [0,1]
        heatmap = cv2.resize(grid, (224, 224))

        return heatmap, target_class


def overlay_heatmap(image_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blends a normalized [0,1] heatmap onto the original RGB image."""
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
