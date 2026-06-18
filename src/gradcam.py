"""
============================================================
src/gradcam.py
Gradient-weighted Class Activation Mapping (Grad-CAM)
============================================================

THEORY
------
Grad-CAM answers: "Which spatial regions did the CNN focus on
to make this prediction?"

Algorithm (Selvaraju et al., 2017):
  1. Forward pass → get class score yᶜ for target class c.
  2. Compute gradients of yᶜ w.r.t. feature maps Aᵏ of the
     chosen conv layer: ∂yᶜ/∂Aᵏ.
  3. Global average pool the gradients → importance weight αᵏᶜ.
     αᵏᶜ = (1/Z) ΣᵢΣⱼ (∂yᶜ/∂Aᵏᵢⱼ)
  4. Weighted sum of feature maps, then ReLU:
     L_Grad-CAM = ReLU(Σₖ αᵏᶜ · Aᵏ)
  5. Upsample to input resolution and overlay as a heatmap.

Why ReLU? We only care about features that positively influence
the class score (negative influences belong to other classes).

Why the last conv layer? It has the best trade-off between
spatial resolution and semantic richness.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Tuple

from src.utils import get_logger, denormalize

logger = get_logger("gradcam")


# ================================================================== #
# Grad-CAM implementation using hooks
# ================================================================== #
class GradCAM:
    """
    Generic Grad-CAM that works with any PyTorch model and target layer.

    Usage:
        cam = GradCAM(model, target_layer=model.backbone.layer4)
        heatmap = cam(input_tensor, class_idx=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks to capture activations and gradients
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ---- Hooks ---- #
    def _save_activation(self, module, input, output):
        """Called during forward pass — stores feature maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Called during backward pass — stores gradients."""
        self._gradients = grad_output[0].detach()

    # ---- Core computation ---- #
    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Grad-CAM heatmap.

        Args:
            x         : input tensor (1, C, H, W) on correct device
            class_idx : target class. If None, uses argmax (predicted class).

        Returns:
            (cam_map, class_idx) : normalized heatmap (H, W) in [0, 1]
                                   and the class index used.
        """
        self.model.eval()

        # Forward
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward for the target class score only
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Gradients/activations not captured. Ensure the target layer "
                "is part of the model's forward pass and gradients are enabled."
            )

        # αᵏ = global average pool of gradients (shape: K)
        grads   = self._gradients            # (1, K, h, w)
        acts    = self._activations          # (1, K, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted sum + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ================================================================== #
# Grad-CAM++ implementation
# ================================================================== #
class GradCAMPlusPlus:
    """Grad-CAM++ — uses second-order gradients for sharper, more precise heatmaps."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks to capture activations and gradients
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ---- Hooks ---- #
    def _save_activation(self, module, input, output):
        """Called during forward pass — stores feature maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Called during backward pass — stores gradients."""
        self._gradients = grad_output[0].detach()

    # ---- Core computation ---- #
    def __call__(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute Grad-CAM++ heatmap.

        Args:
            x         : input tensor (1, C, H, W) on correct device
            class_idx : target class. If None, uses argmax (predicted class).

        Returns:
            (cam_map, class_idx) : normalized heatmap (H, W) in [0, 1]
                                   and the class index used.
        """
        self.model.eval()

        # Forward
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward for the target class score only
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Gradients/activations not captured. Ensure the target layer "
                "is part of the model's forward pass and gradients are enabled."
            )

        grads = self._gradients    # (1, K, h, w)
        acts  = self._activations  # (1, K, h, w)

        # Grad-CAM++ weights:
        #   alpha_k = relu(grad) / (2*grad^2 + sum(acts * grad^3) + eps)
        grad_sq  = grads ** 2
        grad_cb  = grads ** 3
        sum_acts_grad_cb = (acts * grad_cb).sum(dim=(2, 3), keepdim=True)
        alpha = torch.relu(grads) / (2 * grad_sq + sum_acts_grad_cb + 1e-8)

        # Weights: global sum of alpha * relu(grad)
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted sum + ReLU
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ================================================================== #
# Visualization helpers
# ================================================================== #
def overlay_heatmap(
    original_img: np.ndarray,   # (H, W, 3) uint8 RGB
    cam_map: np.ndarray,        # (h, w) float in [0, 1]
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Resize cam_map to match original image and overlay as a colored heatmap.

    Returns:
        overlay : (H, W, 3) uint8 RGB
    """
    H, W = original_img.shape[:2]
    heatmap = cv2.resize(cam_map, (W, H))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return overlay


def visualize_gradcam_batch(
    model: nn.Module,
    images: torch.Tensor,        # (N, C, H, W)
    labels: torch.Tensor,        # (N,)
    target_layer: nn.Module,
    class_names: list,
    cfg: dict,
    num_samples: int = 10,
    device: torch.device = None,
):
    """
    Generate and save Grad-CAM visualizations for a batch of images.

    Saves individual PNG files and a summary grid.
    """
    save_dir = Path(cfg["paths"]["gradcam_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    alpha     = cfg["explainability"]["gradcam"]["alpha"]
    n         = min(num_samples, images.shape[0])
    cam       = GradCAM(model, target_layer)
    cam_pp    = GradCAMPlusPlus(model, target_layer)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]   # ensure 2D

    for i in range(n):
        x     = images[i].unsqueeze(0).to(device)
        label = labels[i].item()

        # Grad-CAM heatmap
        cam_map, pred_idx = cam(x)

        # Grad-CAM++ heatmap (use same predicted class for comparability)
        cam_pp_map, _ = cam_pp(x, class_idx=pred_idx)

        # Compute confidence from the logits already produced inside GradCAM
        with torch.no_grad():
            pred_conf = torch.softmax(model(x), dim=1)[0, pred_idx].item()

        # Denormalize original image
        orig_img = denormalize(images[i])

        # Overlays
        overlay    = overlay_heatmap(orig_img, cam_map, alpha=alpha)
        overlay_pp = overlay_heatmap(orig_img, cam_pp_map, alpha=alpha)

        # Plot — 4 columns: Original | Grad-CAM | Grad-CAM++ | Heatmap++
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original\nTrue: {class_names[label]}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"Grad-CAM\nPred: {class_names[pred_idx]} ({pred_conf:.2%})", fontsize=9
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay_pp)
        axes[i, 2].set_title(
            f"Grad-CAM++\nPred: {class_names[pred_idx]} ({pred_conf:.2%})", fontsize=9
        )
        axes[i, 2].axis("off")

        axes[i, 3].imshow(cam_pp_map, cmap="jet")
        axes[i, 3].set_title("Heatmap++", fontsize=9)
        axes[i, 3].axis("off")

        # Save individual
        ind_path = save_dir / f"gradcam_{i:03d}_{class_names[pred_idx]}.png"
        cv2.imwrite(
            str(ind_path),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

    plt.suptitle("Grad-CAM Explanations", fontsize=16, fontweight="bold")
    plt.tight_layout()
    grid_path = save_dir / "gradcam_grid.png"
    plt.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close()

    cam.remove_hooks()
    cam_pp.remove_hooks()
    logger.info(f"Grad-CAM grid saved → {grid_path}")
