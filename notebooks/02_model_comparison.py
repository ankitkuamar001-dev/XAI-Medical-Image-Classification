"""
============================================================
notebooks/02_model_comparison.py
Compare all model architectures side-by-side
============================================================
"""

# %% [markdown]
# # 02 — Model Comparison
# Side-by-side comparison of all model architectures.
# Each model needs its OWN checkpoint to load correctly.
# If a checkpoint doesn't exist, the model runs with random/pretrained weights.

# %% Setup
import sys
import os
import time
sys.path.insert(0, "..")

import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_config, get_device, set_seed, load_checkpoint
from src.model import build_model, MODEL_REGISTRY
from src.dataset import build_dataloaders
from src.evaluate import evaluate

set_seed(42)
cfg    = load_config("../config.yaml")
device = get_device(cfg["project"]["device"])

_, _, test_loader = build_dataloaders(cfg)

# %% Evaluate all models
results = {}

# Each model needs its own checkpoint. Convention:
#   models/saved_models/{model_name}_best.pth   OR
#   models/saved_models/best_model.pth          (fallback for the default model)
for model_name in MODEL_REGISTRY.keys():
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*50}")

    cfg["training"]["model"]    = model_name
    cfg["training"]["pretrained"] = (model_name != "custom_cnn")

    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    # Try model-specific checkpoint first, then fall back to generic
    model_ckpt = f"../models/saved_models/{model_name}_best.pth"
    generic_ckpt = "../models/saved_models/best_model.pth"

    loaded = False
    if os.path.exists(model_ckpt):
        try:
            load_checkpoint(model_ckpt, model, device=device)
            loaded = True
            print(f"  Loaded: {model_ckpt}")
        except Exception as e:
            print(f"  Failed to load {model_ckpt}: {e}")

    if not loaded and os.path.exists(generic_ckpt):
        try:
            load_checkpoint(generic_ckpt, model, device=device)
            loaded = True
            print(f"  Loaded: {generic_ckpt}")
        except RuntimeError as e:
            # state_dict mismatch — different architecture
            print(f"  Skipping generic checkpoint (architecture mismatch): {model_name}")

    if not loaded:
        print(f"  ⚠ No compatible checkpoint found. Using pretrained/random weights.")

    metrics = evaluate(model, test_loader, cfg, device, split=f"test_{model_name}")

    # Benchmark inference speed (FPS)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(dummy)
    t0 = time.time()
    n_iters = 50
    for _ in range(n_iters):
        with torch.no_grad():
            model(dummy)
    fps = n_iters / (time.time() - t0)

    metrics["params_M"] = round(num_params / 1e6, 1)
    metrics["fps"] = round(fps, 1)
    results[model_name] = metrics
    print(f"  Results: acc={metrics['accuracy']:.4f}, fps={fps:.1f}")

# %% Comparison table
df = pd.DataFrame(results).T

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)
print(df.to_markdown())

# %% Save comparison table as CSV
df.to_csv("../outputs/plots/model_comparison.csv")
print(f"\nSaved → ../outputs/plots/model_comparison.csv")

# %% Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

# Metrics bar chart
ax = axes[0]
metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"] if c in df.columns]
df[metric_cols].plot(
    kind="bar", ax=ax, colormap="viridis", edgecolor="none"
)
ax.set_title("Test Metrics")
ax.set_ylabel("Score")
ax.set_xticklabels(df.index, rotation=15, ha="right")
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

# Speed vs Params scatter
ax = axes[1]
for i, (name, row) in enumerate(df.iterrows()):
    ax.scatter(row["params_M"], row["fps"], s=100, zorder=3)
    ax.annotate(name, (row["params_M"], row["fps"]),
                textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("Inference Speed (FPS)")
ax.set_title("Speed vs Size")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../outputs/plots/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → ../outputs/plots/model_comparison.png")
