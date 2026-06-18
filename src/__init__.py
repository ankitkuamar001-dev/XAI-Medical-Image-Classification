"""
============================================================
src/__init__.py
Explainable Medical Image Classification — Package Exports
============================================================
"""

__all__ = [
    "load_config", "set_seed", "get_device", "make_dirs",
    "save_checkpoint", "load_checkpoint", "denormalize", "get_logger",
    "build_model", "MODEL_REGISTRY",
    "build_dataloaders", "build_train_transforms", "build_val_transforms",
    "train_model", "plot_training_curves",
    "evaluate",
    "GradCAM", "GradCAMPlusPlus", "overlay_heatmap", "visualize_gradcam_batch",
    "compute_shap_values", "visualize_shap", "shap_summary_bar",
]


def __getattr__(name):
    """Lazy imports to avoid loading heavy dependencies at package import time."""
    if name in ("load_config", "set_seed", "get_device", "make_dirs",
                "save_checkpoint", "load_checkpoint", "denormalize", "get_logger"):
        from src import utils
        return getattr(utils, name)
    if name in ("build_model", "MODEL_REGISTRY"):
        from src import model
        return getattr(model, name)
    if name in ("build_dataloaders", "build_train_transforms", "build_val_transforms"):
        from src import dataset
        return getattr(dataset, name)
    if name in ("train_model", "plot_training_curves"):
        from src import train
        return getattr(train, name)
    if name == "evaluate":
        from src import evaluate as eval_mod
        return eval_mod.evaluate
    if name in ("GradCAM", "GradCAMPlusPlus", "overlay_heatmap", "visualize_gradcam_batch"):
        from src import gradcam
        return getattr(gradcam, name)
    if name in ("compute_shap_values", "visualize_shap", "shap_summary_bar"):
        from src import shap_explain
        return getattr(shap_explain, name)
    if name == "mc_dropout_predict":
        from src import uncertainty
        return uncertainty.mc_dropout_predict
    raise AttributeError(f"module 'src' has no attribute {name!r}")
