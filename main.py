"""
============================================================
main.py
Entry point — orchestrates the full pipeline:
  1. Setup (seed, device, dirs)
  2. Data loading
  3. Model building
  4. Training
  5. Evaluation
  6. Grad-CAM + Grad-CAM++ explanations
  7. SHAP explanations
  8. MC Dropout uncertainty estimation
============================================================
Run:
    python main.py                       # full pipeline
    python main.py --mode train          # train only
    python main.py --mode eval           # eval only
    python main.py --mode explain        # XAI only
    python main.py --model resnet50      # override model
    python main.py --model_path path.pth # custom checkpoint
    python main.py --resume              # resume from checkpoint
    python main.py --config config.yaml  # custom config
"""

import argparse
import time
import torch
from pathlib import Path

from src.utils import load_config, set_seed, get_device, make_dirs, load_checkpoint, get_logger
from src.dataset import build_dataloaders
from src.model import build_model
from src.train import train_model, plot_training_curves
from src.evaluate import evaluate
from src.gradcam import GradCAM, visualize_gradcam_batch
from src.shap_explain import compute_shap_values, visualize_shap, shap_summary_bar
from src.uncertainty import mc_dropout_predict, plot_uncertainty

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="XAI Medical Image Classification")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "train", "eval", "explain"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model architecture (e.g., resnet50, densenet121, efficientnet_b0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to a specific model checkpoint (overrides config best_model)",
    )
    return parser.parse_args()


# ================================================================== #
# Pipeline stages
# ================================================================== #

def stage_train(model, train_loader, val_loader, cfg, device):
    """Stage 1: Training."""
    logger.info("=" * 60)
    logger.info("  STAGE: TRAINING")
    logger.info("=" * 60)

    class_weights = train_loader.dataset.get_class_weights()
    history = train_model(
        model, train_loader, val_loader, cfg, device,
        class_weights=class_weights,
    )
    plot_training_curves(history, save_dir=cfg["paths"]["plots_dir"])
    return history


def stage_eval(model, test_loader, cfg, device):
    """Stage 2: Evaluation on test set."""
    logger.info("=" * 60)
    logger.info("  STAGE: EVALUATION")
    logger.info("=" * 60)

    # Load best weights
    best_model_path = cfg["paths"]["best_model"]
    if Path(best_model_path).exists():
        load_checkpoint(best_model_path, model, device=device)

    metrics = evaluate(model, test_loader, cfg, device, split="test")
    return metrics


def stage_explain(model, train_loader, test_loader, cfg, device):
    """Stage 3: Explainability (Grad-CAM + Grad-CAM++ + SHAP)."""
    logger.info("=" * 60)
    logger.info("  STAGE: EXPLAINABILITY")
    logger.info("=" * 60)

    # Load best weights
    best_model_path = cfg["paths"]["best_model"]
    if Path(best_model_path).exists():
        load_checkpoint(best_model_path, model, device=device)
    model.eval()

    # --- Grab a batch of test images ---
    test_images, test_labels = next(iter(test_loader))

    # ---- Grad-CAM + Grad-CAM++ ---- #
    logger.info("[Grad-CAM + Grad-CAM++]")
    try:
        target_layer = model.get_gradcam_layer() \
            if hasattr(model, "get_gradcam_layer") \
            else list(model.children())[-2]

        visualize_gradcam_batch(
            model=model,
            images=test_images,
            labels=test_labels,
            target_layer=target_layer,
            class_names=cfg["dataset"]["class_names"],
            cfg=cfg,
            num_samples=cfg["explainability"]["gradcam"]["num_samples"],
            device=device,
        )
        logger.info("  Grad-CAM + Grad-CAM++ visualizations saved.")
    except Exception as e:
        logger.error(f"  Grad-CAM failed: {e}")

    # ---- SHAP ---- #
    logger.info("[SHAP DeepExplainer]")
    try:
        shap_cfg     = cfg["explainability"]["shap"]
        num_bg       = shap_cfg["background_samples"]
        num_test     = shap_cfg["test_samples"]

        shap_values = compute_shap_values(
            model=model,
            background_loader=train_loader,
            test_images=test_images[:num_test],
            device=device,
            num_background=num_bg,
        )

        visualize_shap(
            shap_values=shap_values,
            test_images=test_images[:num_test],
            class_names=cfg["dataset"]["class_names"],
            cfg=cfg,
            num_samples=num_test,
        )

        shap_summary_bar(
            shap_values=shap_values,
            class_names=cfg["dataset"]["class_names"],
            save_dir=cfg["paths"]["shap_dir"],
        )
        logger.info("  SHAP visualizations saved.")
    except Exception as e:
        logger.error(f"  SHAP failed: {e}")

    # ---- MC Dropout Uncertainty ---- #
    logger.info("[MC Dropout Uncertainty]")
    try:
        n_passes = cfg.get("explainability", {}).get("uncertainty", {}).get(
            "mc_dropout_passes", 30
        )
        class_names = cfg["dataset"]["class_names"]
        num_unc = min(5, test_images.shape[0])

        for i in range(num_unc):
            result = mc_dropout_predict(
                model, test_images[i], device,
                n_forward=n_passes, class_names=class_names,
            )
            save_path = f"{cfg['paths']['plots_dir']}/uncertainty_sample_{i:03d}.png"
            plot_uncertainty(result, class_names, save_path=save_path)

        logger.info(f"  Uncertainty plots saved ({num_unc} samples).")
    except Exception as e:
        logger.error(f"  Uncertainty estimation failed: {e}")


# ================================================================== #
# Main
# ================================================================== #

def main():
    pipeline_start = time.time()
    args = parse_args()

    # ---- Setup ----
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        return
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    make_dirs(cfg)

    # Override model from CLI if specified
    if args.model:
        logger.info(f"CLI override: model = {args.model}")
        cfg["training"]["model"] = args.model

    # Override model path from CLI if specified
    if args.model_path:
        logger.info(f"CLI override: model_path = {args.model_path}")
        cfg["paths"]["best_model"] = args.model_path

    # ---- Data ----
    logger.info("[Data] Building DataLoaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ---- Model ----
    logger.info("[Model] Building model …")
    model = build_model(cfg).to(device)

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_path = cfg["paths"]["checkpoint"]
        if Path(ckpt_path).exists():
            load_checkpoint(ckpt_path, model, device=device)
            logger.info(f"Resumed from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, starting fresh.")

    # ---- Run pipeline ----
    mode = args.mode

    if mode in ("all", "train"):
        stage_train(model, train_loader, val_loader, cfg, device)

    if mode in ("all", "eval"):
        stage_eval(model, test_loader, cfg, device)

    if mode in ("all", "explain"):
        stage_explain(model, train_loader, test_loader, cfg, device)

    elapsed = time.time() - pipeline_start
    logger.info(f"\n✅ Pipeline complete in {elapsed:.1f}s. Check outputs/ for results.")


if __name__ == "__main__":
    main()
