"""
============================================================
notebooks/01_EDA.py  (run as Jupyter cell-by-cell or convert)
Exploratory Data Analysis — Chest X-Ray Dataset
============================================================
Convert to notebook: jupytext --to notebook notebooks/01_EDA.py
"""

# %% [markdown]
# # 01 — Exploratory Data Analysis
# **Dataset**: Chest X-Ray Images (Pneumonia)
#
# Goals:
# 1. Understand class distribution (imbalance check)
# 2. Visualize sample images per class
# 3. Analyze pixel intensity distributions
# 4. Check image dimensions

# %% Setup
import sys
sys.path.insert(0, "..")

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

plt.style.use("dark_background")

DATA_DIR = Path("../data/raw/chest_xray")
SPLITS   = ["train", "val", "test"]
CLASSES  = ["NORMAL", "PNEUMONIA"]

# %% [markdown]
# ## 1. Class Distribution

# %%
counts = defaultdict(dict)
for split in SPLITS:
    for cls in CLASSES:
        d = DATA_DIR / split / cls
        # Filter by image extension to avoid counting .DS_Store, etc.
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        counts[split][cls] = len([f for f in d.glob("*") if f.suffix.lower() in img_exts]) if d.exists() else 0

print("Image counts per split:")
for split in SPLITS:
    total = sum(counts[split].values())
    print(f"  {split:6s}: {counts[split]} | Total: {total}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors    = {"NORMAL": "#3fb950", "PNEUMONIA": "#f85149"}

for ax, split in zip(axes, SPLITS):
    vals = [counts[split][c] for c in CLASSES]
    bars = ax.bar(CLASSES, vals, color=[colors[c] for c in CLASSES], edgecolor="none")
    ax.bar_label(bars, fmt="%d", padding=4)
    ax.set_title(f"{split.capitalize()} Split", fontsize=13, fontweight="bold")
    ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.2)

plt.suptitle("Class Distribution per Split", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/plots/eda_class_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Sample Images per Class

# %%
fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(2, 8, hspace=0.3, wspace=0.05)

for row_idx, cls in enumerate(CLASSES):
    cls_dir   = DATA_DIR / "train" / cls
    img_paths = list(cls_dir.glob("*.jpeg"))[:8]
    for col_idx, img_path in enumerate(img_paths):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        img = Image.open(img_path).convert("L")
        ax.imshow(img, cmap="bone")
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel(cls, fontsize=11, fontweight="bold",
                          color=colors[cls], labelpad=40)

plt.suptitle("Sample Images — NORMAL vs PNEUMONIA", fontsize=15, fontweight="bold")
plt.savefig("../outputs/plots/eda_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Pixel Intensity Distributions
#
# Pneumonia X-rays typically show higher pixel density in lung regions.

# %%
def sample_intensities(cls_dir: Path, n: int = 100) -> np.ndarray:
    """Sample mean pixel intensity from n images."""
    imgs  = list(cls_dir.glob("*.jpeg"))[:n]
    means = []
    for p in imgs:
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
        means.append(arr.mean())
    return np.array(means)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, split in zip(axes, ["train", "test"]):
    for cls in CLASSES:
        d = DATA_DIR / split / cls
        if d.exists():
            ints = sample_intensities(d)
            ax.hist(ints, bins=30, alpha=0.6, label=cls, color=colors[cls],
                    density=True, edgecolor="none")
    ax.set_title(f"Mean Pixel Intensity ({split})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Pixel Intensity (0–255)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.2)

plt.suptitle("Pixel Intensity Distribution", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/plots/eda_intensity_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Image Dimension Analysis

# %%
dims = []
for p in list((DATA_DIR / "train" / "NORMAL").glob("*.jpeg"))[:50]:
    img = Image.open(p)
    dims.append(img.size)   # (W, H)

widths  = [d[0] for d in dims]
heights = [d[1] for d in dims]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, vals, label in zip(axes, [widths, heights], ["Width", "Height"]):
    ax.hist(vals, bins=20, color="#58a6ff", edgecolor="none", alpha=0.8)
    ax.set_title(f"Image {label} Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{label} (pixels)")
    ax.set_ylabel("Count")
    ax.axvline(np.mean(vals), color="orange", ls="--", label=f"Mean: {np.mean(vals):.0f}")
    ax.legend()
    ax.grid(alpha=0.2)

plt.suptitle("Image Dimension Distribution (NORMAL, train split)", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/plots/eda_dimensions.png", dpi=150)
plt.show()

print(f"\nMean width: {np.mean(widths):.0f} | Mean height: {np.mean(heights):.0f}")
print("→ Images are resized to 224×224 during preprocessing.")
