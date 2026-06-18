"""
============================================================
streamlit_app/app.py
Web Application — Upload X-ray image → CNN Prediction +
                  Grad-CAM / Grad-CAM++ heatmap + SHAP +
                  MC Dropout Uncertainty
============================================================

Run:
    streamlit run streamlit_app/app.py

Features:
  - Drag & drop medical image upload
  - Try sample images from test set
  - Real-time CNN inference
  - Grad-CAM AND Grad-CAM++ heatmap overlays
  - SHAP pixel attribution map
  - Monte Carlo Dropout uncertainty estimation
  - Prediction confidence + class probabilities
  - Enhanced report download (with metrics)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from src.utils import load_config, get_device, load_checkpoint, denormalize
from src.model import build_model
from src.dataset import build_val_transforms
from src.gradcam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from src.uncertainty import mc_dropout_predict


# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="XAI Medical Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Custom CSS — dark theme + glassmorphism cards + animations
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: #0d1117;
        color: #e6edf3;
    }

    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    h1 { background: linear-gradient(90deg, #58a6ff, #bc8cff);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700; font-size: 2.8rem; letter-spacing: -0.5px;}

    .metric-card {
        background: rgba(22,27,34,0.6);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 16px; padding: 24px; margin: 12px 0;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(88,166,255,0.1);
    }

    .pred-normal   { color: #3fb950; font-weight: 700; font-size: 2rem; margin-bottom: 0;}
    .pred-pneumonia{ color: #f85149; font-weight: 700; font-size: 2rem; margin-bottom: 0;}
    .confidence    { color: #8b949e; font-size: 1.1rem; margin-top: -5px;}

    .uncertainty-low  { color: #3fb950; font-weight: 600; }
    .uncertainty-mid  { color: #d29922; font-weight: 600; }
    .uncertainty-high { color: #f85149; font-weight: 600; }

    .stProgress > div > div { background: linear-gradient(90deg, #58a6ff, #bc8cff) !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; white-space: pre-wrap; 
        background-color: rgba(22,27,34,0.5); border-radius: 8px 8px 0 0;
        border: 1px solid rgba(88,166,255,0.1); border-bottom: none;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(88,166,255,0.15); border-color: #58a6ff; }

    .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; border-radius: 8px; border: none;
        padding: 0.6rem 1.4rem; font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(46,160,67,0.4); }

    .stFileUploader { background: rgba(22,27,34,0.6); border-radius: 16px; padding: 20px; border: 1px dashed #30363d;}
    .stSidebar { background: #161b22 !important; border-right: 1px solid #30363d;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================== #
# Model loading (cached)
# ================================================================== #
@st.cache_resource
def load_model_and_cfg():
    cfg    = load_config("config.yaml")
    device = get_device(cfg["project"]["device"])
    model  = build_model(cfg).to(device)

    best_path = cfg["paths"]["best_model"]
    if Path(best_path).exists():
        load_checkpoint(best_path, model, device=device)
        return model, cfg, device, True
    return model, cfg, device, False


# ================================================================== #
# Inference + explainability
# ================================================================== #
def predict(model, img_tensor, cfg, device):
    """Run inference. Returns (pred_idx, probs)."""
    model.eval()
    with torch.no_grad():
        x      = img_tensor.unsqueeze(0).to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    return pred_idx, probs


def get_gradcam(model, img_tensor, pred_idx, cfg, device):
    """Return Grad-CAM overlay and raw heatmap."""
    target_layer = model.get_gradcam_layer() if hasattr(model, "get_gradcam_layer") \
                   else list(model.children())[-2]
    cam     = GradCAM(model, target_layer)
    x       = img_tensor.unsqueeze(0).to(device)
    cam_map, _ = cam(x, class_idx=pred_idx)
    cam.remove_hooks()

    orig_img = denormalize(img_tensor)
    alpha    = cfg["explainability"]["gradcam"]["alpha"]
    overlay  = overlay_heatmap(orig_img, cam_map, alpha=alpha)
    return overlay, cam_map


def get_gradcam_pp(model, img_tensor, pred_idx, cfg, device):
    """Return Grad-CAM++ overlay and raw heatmap."""
    target_layer = model.get_gradcam_layer() if hasattr(model, "get_gradcam_layer") \
                   else list(model.children())[-2]
    cam     = GradCAMPlusPlus(model, target_layer)
    x       = img_tensor.unsqueeze(0).to(device)
    cam_map, _ = cam(x, class_idx=pred_idx)
    cam.remove_hooks()

    orig_img = denormalize(img_tensor)
    alpha    = cfg["explainability"]["gradcam"]["alpha"]
    overlay  = overlay_heatmap(orig_img, cam_map, alpha=alpha)
    return overlay, cam_map


def get_shap(model, img_tensor, device, num_bg=20):
    """
    Compute a lightweight SHAP explanation (GradientExplainer — faster for web app).
    Uses diverse Gaussian noise backgrounds for better statistical validity.
    """
    import shap
    model.eval()
    # Create diverse background: random Gaussian noise centered around ImageNet stats
    bg = torch.randn(num_bg, *img_tensor.shape).to(device) * 0.2

    explainer   = shap.GradientExplainer(model, bg)
    shap_values = explainer.shap_values(img_tensor.unsqueeze(0).to(device))
    # Normalize for newer SHAP versions that return ndarray instead of list
    if isinstance(shap_values, np.ndarray):
        shap_values = [shap_values[..., i] for i in range(shap_values.shape[-1])]
    return shap_values


# ================================================================== #
# UI Sidebar & Sampling
# ================================================================== #
def render_sidebar(cfg, model_loaded):
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown(f"**Model:** `{cfg['training']['model']}`")
        st.markdown(f"**Classes:** {', '.join(cfg['dataset']['class_names'])}")
        
        status = "✅ Loaded" if model_loaded else "⚠️ Untrained"
        st.markdown(f"**Weights:** {status}")
        st.markdown("---")

        st.markdown("### 🖼️ Try a Sample")
        st.markdown("Don't have an X-ray? Pick one below:")
        
        sample_img = None
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Pneumonia 1"):
                sample_img = "data/raw/chest_xray/test/PNEUMONIA/person147_bacteria_706.jpeg"
            if st.button("Pneumonia 2"):
                sample_img = "data/raw/chest_xray/test/PNEUMONIA/person100_bacteria_482.jpeg"
        with col2:
            if st.button("Normal 1"):
                sample_img = "data/raw/chest_xray/test/NORMAL/IM-0031-0001.jpeg"
            if st.button("Normal 2"):
                sample_img = "data/raw/chest_xray/test/NORMAL/NORMAL2-IM-0272-0001.jpeg"
                
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown(
            "This app uses deep learning to classify chest X-rays and explain "
            "decisions with **Grad-CAM**, **Grad-CAM++**, **SHAP**, and "
            "**MC Dropout** uncertainty estimation."
        )
        st.markdown("---")
        st.markdown(
            "### 🏗️ Architecture\n"
            "| Model | Params |\n"
            "|-------|--------|\n"
            "| Custom CNN | ~1.5M |\n"
            "| ResNet-50 | ~25M |\n"
            "| DenseNet-121 | ~8M |\n"
            "| VGG-16 | ~138M |\n"
            "| EfficientNet-B0 | ~5.3M |"
        )
        return sample_img


def main():
    st.markdown("# 🫁 XAI Medical Image Classifier")
    st.markdown(
        "Upload a **chest X-ray** or select a sample from the sidebar to get a CNN prediction with "
        "**Grad-CAM**, **Grad-CAM++**, **SHAP**, and **uncertainty** explanations."
    )

    model, cfg, device, model_loaded = load_model_and_cfg()
    sample_img_path = render_sidebar(cfg, model_loaded)

    # ---- Upload or Sample ----
    uploaded = st.file_uploader(
        "Drop a chest X-ray here (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="xray_upload",
    )

    pil_img = None
    if uploaded is not None:
        pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
    elif sample_img_path and Path(sample_img_path).exists():
        pil_img = Image.open(sample_img_path).convert("RGB")

    if pil_img is None:
        st.info("👆 Upload an X-ray image or click a sample button in the sidebar to begin analysis.")
        return

    transform = build_val_transforms(cfg["dataset"]["image_size"][0])
    img_tensor = transform(pil_img)
    class_names = cfg["dataset"]["class_names"]

    # ---- Run prediction + uncertainty ----
    with st.spinner("🧠 Analyzing image..."):
        pred_idx, probs = predict(model, img_tensor, cfg, device)
        # MC Dropout uncertainty
        n_passes = cfg.get("explainability", {}).get("uncertainty", {}).get("mc_dropout_passes", 30)
        unc_result = mc_dropout_predict(
            model, img_tensor, device,
            n_forward=n_passes, class_names=class_names
        )

    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx]
    uncertainty = unc_result["uncertainty"]
    entropy = unc_result["entropy"]

    # Uncertainty level classification
    if uncertainty < 0.05:
        unc_level, unc_css = "Low", "uncertainty-low"
    elif uncertainty < 0.15:
        unc_level, unc_css = "Moderate", "uncertainty-mid"
    else:
        unc_level, unc_css = "High", "uncertainty-high"

    # ================================================================ #
    # Results layout
    # ================================================================ #
    st.markdown("---")
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### 🖼️ Input Image")
        st.image(pil_img, use_container_width=True, caption="Analyzed X-ray")

    with col2:
        st.markdown("### 📊 Prediction Result")
        tag_cls = "pred-normal" if pred_idx == 0 else "pred-pneumonia"
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="{tag_cls}">{pred_label}</p>'
            f'<p class="confidence">Confidence: {confidence:.2%}</p>'
            f'<p class="{unc_css}">Uncertainty: {unc_level} (±{uncertainty:.4f}, entropy: {entropy:.3f} bits)</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>**Class Probabilities**", unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(zip(class_names, probs)):
            st.markdown(f"`{cls}`: {prob:.1%}")
            st.progress(float(prob))
            
    st.markdown("---")

    # ================================================================ #
    # Explainability Tabs
    # ================================================================ #
    st.markdown("### 🕵️ Model Interpretability")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔥 Grad-CAM", "🔥+ Grad-CAM++", "🔮 SHAP Analysis",
        "📊 Uncertainty", "📝 Download Report"
    ])
    
    with tab1:
        st.markdown(
            "**Grad-CAM** highlights the spatial regions driving the prediction. "
            "Warmer colors (red/yellow) indicate higher activation."
        )
        with st.spinner("Computing Grad-CAM …"):
            try:
                overlay, cam_map = get_gradcam(model, img_tensor, pred_idx, cfg, device)
                c1, c2 = st.columns(2)
                c1.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
                c2.image(plt.cm.jet(cam_map)[:, :, :3], caption="Raw Heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM error: {e}")

    with tab2:
        st.markdown(
            "**Grad-CAM++** uses second-order gradients for sharper, more precise heatmaps. "
            "Compare with standard Grad-CAM to see which regions are highlighted more precisely."
        )
        with st.spinner("Computing Grad-CAM++ …"):
            try:
                overlay_pp, cam_map_pp = get_gradcam_pp(model, img_tensor, pred_idx, cfg, device)
                c1, c2 = st.columns(2)
                c1.image(overlay_pp, caption="Grad-CAM++ Overlay", use_container_width=True)
                c2.image(plt.cm.jet(cam_map_pp)[:, :, :3], caption="Grad-CAM++ Heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM++ error: {e}")

    with tab3:
        st.markdown(
            "**SHAP** attributes each pixel's signed contribution to the prediction. "
            "Red = pushes towards class, Blue = pushes away from class."
        )
        with st.spinner("Computing SHAP values (may take ~30s) …"):
            try:
                shap_values = get_shap(model, img_tensor, device, num_bg=20)
                fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(4 * (len(class_names) + 1), 3.5))
                fig.patch.set_facecolor("#0d1117")

                orig_arr = np.array(pil_img).astype(np.float32) / 255.0
                axes[0].imshow(orig_arr)
                axes[0].set_title("Original", color="white")
                axes[0].axis("off")

                for ci, cls_name in enumerate(class_names):
                    sv = shap_values[ci][0]        # (C, H, W)
                    sv = np.transpose(sv, (1, 2, 0))  # (H, W, C)
                    saliency = np.abs(sv).mean(axis=-1)
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                    im = axes[ci + 1].imshow(saliency, cmap="RdBu_r", vmin=0, vmax=1)
                    axes[ci + 1].set_title(f"SHAP — {cls_name}", color="white")
                    axes[ci + 1].axis("off")
                    plt.colorbar(im, ax=axes[ci + 1], fraction=0.046, pad=0.04)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"SHAP error: {e}")

    with tab4:
        st.markdown(
            "**Monte Carlo Dropout** runs multiple stochastic forward passes to estimate "
            "how *certain* the model is about its prediction. High uncertainty flags cases "
            "that may need expert review."
        )
        try:
            from src.uncertainty import plot_uncertainty
            fig = plot_uncertainty(unc_result, class_names)
            st.pyplot(fig)
            plt.close()

            # Interpretation text
            if unc_level == "Low":
                st.success(
                    f"✅ **Low uncertainty** — the model is highly confident in its "
                    f"prediction of **{pred_label}** ({confidence:.1%} ± {uncertainty:.4f})."
                )
            elif unc_level == "Moderate":
                st.warning(
                    f"⚠️ **Moderate uncertainty** — the model's confidence varies across "
                    f"stochastic passes. Consider clinical review."
                )
            else:
                st.error(
                    f"🚨 **High uncertainty** — the model is not confident. This case should "
                    f"be reviewed by a radiologist."
                )
        except Exception as e:
            st.error(f"Uncertainty error: {e}")
                
    with tab5:
        st.markdown("#### 📥 Export Results")
        report_text = (
            f"XAI Medical Classifier Report\n"
            f"=============================\n"
            f"Model: {cfg['training']['model']}\n"
            f"Prediction: {pred_label}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Uncertainty: {unc_level} (±{uncertainty:.4f})\n"
            f"Entropy: {entropy:.3f} bits\n\n"
            f"MC Dropout Passes: {n_passes}\n\n"
            f"Probabilities:\n"
        )
        for i, (cls, prob) in enumerate(zip(class_names, probs)):
            mean_p = unc_result["mean_probs"][i]
            std_p = unc_result["std_probs"][i]
            report_text += f"  - {cls}: {prob:.2%} (MC: {mean_p:.4f} ± {std_p:.4f})\n"

        report_text += (
            f"\nInterpretation:\n"
            f"  Uncertainty Level: {unc_level}\n"
        )
        if unc_level == "High":
            report_text += "  ⚠ This case should be reviewed by a radiologist.\n"

        st.download_button(
            label="Download Text Report",
            data=report_text,
            file_name="xai_report.txt",
            mime="text/plain",
        )

    # ================================================================ #
    # Interpretation note
    # ================================================================ #
    st.markdown("---")
    st.markdown(
        "> ⚠️ **Clinical Disclaimer**: This tool is for research/educational purposes only. "
        "Always rely on licensed radiologists for clinical diagnosis."
    )


if __name__ == "__main__":
    main()
