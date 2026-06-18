"""
============================================================
src/dataset.py
Data loading, preprocessing, and augmentation pipeline
============================================================

Dataset used: Chest X-Ray Images (Pneumonia)
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Directory structure expected after download:
  data/raw/
    chest_xray/
      train/
        NORMAL/   *.jpeg
        PNEUMONIA/ *.jpeg
      val/
        NORMAL/
        PNEUMONIA/
      test/
        NORMAL/
        PNEUMONIA/
"""


from pathlib import Path
from typing import Tuple, Optional, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


from src.utils import get_logger

logger = get_logger("dataset")


# ------------------------------------------------------------------ #
# ImageNet normalization stats (works well for medical images too)
# ------------------------------------------------------------------ #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ------------------------------------------------------------------ #
# Transform builders
# ------------------------------------------------------------------ #
def build_train_transforms(image_size: int = 224,
                           augment_cfg: dict = None) -> T.Compose:
    """
    Training transforms: augmentation + normalization.

    Augmentation rationale:
      • Horizontal flip — X-rays can be mirrored without losing clinical meaning.
      • Rotation (±15°) — Slight rotation mimics patient positioning variation.
      • Color jitter — Brightness/contrast vary across scanners.
      • Random erasing — Forces the model to use distributed features (robustness).
    """
    aug = augment_cfg or {}

    transforms = [
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5 if aug.get("horizontal_flip", True) else 0),
        T.RandomRotation(degrees=aug.get("rotation_degrees", 15)),
        T.ColorJitter(
            brightness=aug.get("brightness_jitter", 0.2),
            contrast=aug.get("contrast_jitter", 0.2),
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if aug.get("random_erasing", True):
        transforms.append(T.RandomErasing(p=0.3, scale=(0.02, 0.1)))

    return T.Compose(transforms)


def build_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Validation/Test transforms: only resize + normalize (no augmentation).
    This ensures evaluation is deterministic.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ------------------------------------------------------------------ #
# Custom Dataset
# ------------------------------------------------------------------ #
class MedicalImageDataset(Dataset):
    """
    Generic medical image dataset.

    Supports any folder structure where each class is a sub-directory:
        root/
          CLASS_A/  image1.jpg, image2.jpg ...
          CLASS_B/  image1.jpg ...
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        root: str,
        class_names: List[str],
        transform: Optional[T.Compose] = None,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Scan root directory and collect (image_path, label) pairs."""
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            label = self.class_to_idx[class_name]
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, label))

        logger.info(
            f"Loaded {len(self.samples)} images from {self.root} "
            f"| Classes: {self.class_names}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        # Open as RGB (handles grayscale X-rays by converting to 3-channel)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for imbalanced datasets.
        Returns a tensor of shape (num_classes,).
        """
        counts = np.zeros(len(self.class_names))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_names)
        return torch.FloatTensor(weights)


# ------------------------------------------------------------------ #
# DataLoader factory
# ------------------------------------------------------------------ #
def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Returns:
        train_loader, val_loader, test_loader
    """
    ds_cfg  = cfg["dataset"]
    aug_cfg = cfg["augmentation"]
    tr_cfg  = cfg["training"]

    raw_dir    = Path(ds_cfg["raw_dir"]) / ds_cfg["name"]
    img_size   = ds_cfg["image_size"][0]   # assume square
    class_names = ds_cfg["class_names"]
    batch_size   = tr_cfg["batch_size"]
    num_workers  = tr_cfg.get("num_workers", 0)

    # Build transforms
    train_tf = build_train_transforms(img_size, aug_cfg)
    val_tf   = build_val_transforms(img_size)

    # Expected Kaggle layout: raw/chest_xray/train | val | test
    train_root = raw_dir / "train"
    val_root   = raw_dir / "val"
    test_root  = raw_dir / "test"

    train_ds = MedicalImageDataset(str(train_root), class_names, train_tf)
    val_ds   = MedicalImageDataset(str(val_root),   class_names, val_tf)
    test_ds  = MedicalImageDataset(str(test_root),  class_names, val_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,        # Avoid batch-norm issues with batch size 1
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"DataLoaders → Train: {len(train_ds)}, "
        f"Val: {len(val_ds)}, Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader
