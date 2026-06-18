# 🫁 Explainable Medical Image Classification using CNN with Grad-CAM and SHAP

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A **production-quality, end-to-end XAI pipeline** for medical image classification with deep learning explainability. Built for research and education.

---

## 🎯 Objectives

1. Build CNN-based medical image classification models (5 architectures)
2. Apply explainability techniques: **Grad-CAM**, **Grad-CAM++**, and **SHAP**
3. Estimate prediction **uncertainty** via Monte Carlo Dropout
4. Provide reproducible, well-documented pipeline with web interface

## 📊 Results (Chest X-Ray Pneumonia)

| Metric | Score |
|--------|-------|
| **Accuracy** | 91% |
| **Precision (Pneumonia)** | 90% |
| **Recall / Sensitivity** | 95% |
| **Specificity** | 84% |
| **F1 Score** | 93% |
| **ROC-AUC** | ~0.96 |

> In medical AI, **Recall > Precision** — missing a pneumonia case (False Negative) is clinically far more dangerous than a false alarm.

---

## 🏗️ Architecture Options

| Model | Params | XAI Quality | Speed |
|-------|--------|-------------|-------|
| Custom CNN | ~1.5M | ⭐⭐⭐ | Fast |
| **ResNet-50** (default) | ~25M | ⭐⭐⭐⭐⭐ | Medium |
| DenseNet-121 | ~8M | ⭐⭐⭐⭐⭐ | Medium |
| VGG-16 | ~138M | ⭐⭐⭐⭐ | Slow |
| EfficientNet-B0 | ~5.3M | ⭐⭐⭐⭐ | Fast |

---

## 📂 Project Structure

```
XAI/
├── config.yaml                 # Central config (dataset, model, training, XAI)
├── main.py                     # Pipeline orchestrator (--mode all/train/eval/explain)
├── download_data.py            # Kaggle dataset downloader
├── requirements.txt            # Dependencies (cleaned, no unused packages)
├── .gitignore                  # Git ignore rules
│
├── src/
│   ├── __init__.py             # Lazy package exports (fast startup)
│   ├── utils.py                # Seed, device, logging, checkpoint helpers
│   ├── dataset.py              # MedicalImageDataset, transforms, DataLoader factory
│   ├── model.py                # 5 architectures + factory (CustomCNN, ResNet50,
│   │                           #   VGG16, EfficientNet-B0, DenseNet-121)
│   ├── train.py                # Training loop, AMP, early stopping, tqdm, TensorBoard
│   ├── evaluate.py             # Metrics (incl. specificity/sensitivity), confusion matrix,
│   │                           #   ROC, Precision-Recall curve
│   ├── gradcam.py              # Grad-CAM + Grad-CAM++ (side-by-side comparison)
│   ├── shap_explain.py         # SHAP DeepExplainer + pixel attribution maps
│   └── uncertainty.py          # Monte Carlo Dropout uncertainty estimation
│
├── streamlit_app/
│   └── app.py                  # Dark-themed web app (5 tabs: Grad-CAM, Grad-CAM++,
│                               #   SHAP, Uncertainty, Report)
│
├── notebooks/
│   ├── 01_EDA.py               # EDA: class distribution, samples, pixel intensity
│   └── 02_model_comparison.py  # Multi-model benchmark with FPS + scatter plot
│
├── XAI_Medical_Colab.ipynb     # Self-contained Google Colab notebook
│
├── models/saved_models/        # Trained model weights (.pth)
├── outputs/                    # Generated plots, heatmaps, SHAP maps
│   ├── plots/                  # Training curves, confusion matrix, ROC, PR, uncertainty
│   ├── gradcam/                # Grad-CAM + Grad-CAM++ side-by-side grids
│   └── shap/                   # SHAP attribution maps
└── data/raw/                   # Dataset (downloaded via Kaggle API)
```

---

## 🚀 Quick Start

### Local Setup
```bash
cd XAI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset (needs Kaggle API key)
python download_data.py

# Run full pipeline
python main.py                           # all stages
python main.py --mode train              # train only
python main.py --mode eval               # evaluate only
python main.py --mode explain            # XAI + uncertainty
python main.py --model densenet121       # override architecture
python main.py --model_path path.pth     # custom checkpoint

# Web App
streamlit run streamlit_app/app.py
```

### Google Colab
Upload `XAI_Medical_Colab.ipynb` to Google Colab and run all cells. GPU recommended.

---

## 🧠 Explainability Techniques

### Grad-CAM / Grad-CAM++
```
yᶜ (class score)
  ↓ backprop
∂yᶜ / ∂Aᵏ  (gradients w.r.t. conv feature maps)
  ↓ global average pool
αᵏᶜ = (1/Z) Σᵢⱼ ∂yᶜ/∂Aᵏᵢⱼ   (importance weights)
  ↓ weighted sum + ReLU
L_Grad-CAM = ReLU(Σₖ αᵏᶜ · Aᵏ)   (heatmap)
```
**Grad-CAM++** uses second-order gradients for sharper, more precise heatmaps — particularly better when multiple instances of a class appear in the image.

### SHAP (Shapley Values)
```
f(x) = φ₀ + Σᵢ φᵢ

where:
  φ₀  = E[f(x)]        (expected prediction over background)
  φᵢ  = SHAP value of pixel i  (signed contribution)
```

### Monte Carlo Dropout Uncertainty
```
Run N forward passes with dropout ON:
  p̂ = (1/N) Σₙ softmax(f_θ(x; mask_n))    (mean prediction)
  σ = std(p̂₁, ..., p̂ₙ)                     (epistemic uncertainty)
  H = -Σ p̂ᵢ log₂(p̂ᵢ)                       (predictive entropy)
```

---

## 🔧 Configuration (`config.yaml`)

```yaml
training:
  model: "resnet50"        # custom_cnn | resnet50 | densenet121 | vgg16 | efficientnet_b0
  pretrained: true
  freeze_backbone: false
  epochs: 30
  batch_size: 32
  learning_rate: 0.0001
  dropout: 0.5             # Classifier head dropout
  num_workers: 0           # DataLoader workers (0 = safest)
  gradient_clip_norm: 1.0  # Max gradient norm
  scheduler: "cosine"      # cosine | step | plateau
  early_stopping_patience: 7
  mixed_precision: true    # AMP — requires CUDA GPU

explainability:
  uncertainty:
    mc_dropout_passes: 30  # Stochastic forward passes
```

---

## 📈 Outputs

After running the pipeline, you'll find:
- `outputs/plots/training_curves.png` — Loss + accuracy per epoch
- `outputs/plots/confusion_matrix.png` — Raw + normalized confusion matrix
- `outputs/plots/roc_curve.png` — ROC with AUC score
- `outputs/plots/precision_recall_curve.png` — PR curve with AP score
- `outputs/plots/metrics.json` — All metrics (incl. specificity/sensitivity)
- `outputs/plots/classification_report.txt` — Detailed classification report
- `outputs/plots/uncertainty_sample_*.png` — MC Dropout uncertainty plots
- `outputs/gradcam/gradcam_grid.png` — Grad-CAM vs Grad-CAM++ side-by-side
- `outputs/shap/shap_sample_*.png` — Per-sample SHAP attribution maps

---

## 🖥️ Streamlit Web App

The web app features **5 explainability tabs**:
1. 🔥 **Grad-CAM** — Spatial heatmap overlay
2. 🔥+ **Grad-CAM++** — Sharper, second-order heatmaps
3. 🔮 **SHAP** — Pixel-level signed attribution
4. 📊 **Uncertainty** — MC Dropout confidence intervals
5. 📝 **Report** — Downloadable analysis report

---

## ⚠️ Limitations

1. **Class imbalance** (3:1 PNEUMONIA/NORMAL) — mitigated by class weights
2. **Binary classification only** — real-world X-rays have 14+ pathologies
3. **SHAP speed** — DeepExplainer is slow (~30s per sample on CPU)
4. **No clinical validation** — for research/education only

## 🔮 Future Work

- [ ] Multi-label (NIH ChestX-ray14 — 14 classes)
- [ ] Vision Transformer (ViT) with attention visualization
- [ ] Confidence calibration (temperature scaling)
- [ ] Docker + FastAPI for production deployment
- [ ] DICOM image format support

---

> ⚠️ **Clinical Disclaimer**: This tool is for research/educational purposes only. Always rely on licensed radiologists for clinical diagnosis.
