# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🫁 Explainable Medical Image Classification
# ### CNN with Grad-CAM, Grad-CAM++, and SHAP — Enhanced Edition
#
# **Pipeline:** Data → Train (ResNet50/DenseNet121) → Evaluate → Grad-CAM/Grad-CAM++ → SHAP
#
# **Enhancements in this version:**
# - DenseNet-121 architecture + Grad-CAM++ for sharper heatmaps
# - Precision-Recall curve (critical for imbalanced medical data)
# - Metrics saved to JSON + classification report to text
# - Fixed AMP API for PyTorch 2.x, SHAP version compatibility
# - tqdm progress bars, gradient clipping in all training paths
# - Robust checkpoint loading with format safety

# %% [markdown]
# ## 0. Environment Setup

# %%
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

# Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    device = torch.device('cuda')
else:
    print("⚠️ No GPU detected — using CPU (training will be slow)")
    device = torch.device('cpu')

# %% [markdown]
# ## 1. Mount Google Drive & Download Dataset

# %%
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/XAI_Medical'
    IN_COLAB = True
except ImportError:
    BASE = '.'
    IN_COLAB = False

os.makedirs(BASE, exist_ok=True)
DATA_DIR  = f'{BASE}/data/chest_xray'
MODEL_DIR = f'{BASE}/models'
OUT_DIR   = f'{BASE}/outputs'

for d in [DATA_DIR, MODEL_DIR, f'{OUT_DIR}/plots', f'{OUT_DIR}/gradcam', f'{OUT_DIR}/shap']:
    os.makedirs(d, exist_ok=True)

print(f"Base directory: {BASE}")

# %%
# Download dataset if needed
if not os.path.exists(f'{DATA_DIR}/train'):
    print("Downloading Chest X-Ray dataset...")
    os.system('pip install -q kaggle')
    os.system(f'kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {BASE}/data --unzip')
    # Fix nested directory if present
    nested = f'{BASE}/data/chest_xray/chest_xray'
    if os.path.exists(nested):
        import shutil
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(DATA_DIR, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)

# Verify
for split in ['train', 'val', 'test']:
    sp = f'{DATA_DIR}/{split}'
    if os.path.exists(sp):
        n = sum(len(files) for _, _, files in os.walk(sp))
        print(f"  {split}: {n} images")

# %% [markdown]
# ## 2. Configuration

# %%
CFG = {
    'data_dir':     DATA_DIR,
    'model_dir':    MODEL_DIR,
    'out_dir':      OUT_DIR,
    'class_names':  ['NORMAL', 'PNEUMONIA'],
    'num_classes':  2,
    'image_size':   224,
    'batch_size':   32,
    'epochs':       15,
    'lr':           1e-4,
    'weight_decay': 1e-4,
    'patience':     7,
    'model_name':   'resnet50',  # resnet50 | densenet121 | efficientnet_b0 | custom_cnn
    'pretrained':   True,
    'freeze_backbone': False,
    'model_path':   f'{MODEL_DIR}/best_model.pth',
    'gradcam_alpha': 0.5,
    'shap_bg':      50,
    'shap_test':    10,
}

print("Configuration:")
for k, v in CFG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 3. Imports & Seed

# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import cv2
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

plt.style.use('dark_background')

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("✅ Seed set, imports loaded")

# %% [markdown]
# ## 4. Dataset & DataLoaders

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize((CFG['image_size'], CFG['image_size'])),
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.3, scale=(0.02, 0.1)),
])

val_tf = T.Compose([
    T.Resize((CFG['image_size'], CFG['image_size'])),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ChestXrayDataset(Dataset):
    VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, root, class_names, transform=None):
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []
        for cls in class_names:
            cls_dir = Path(root) / cls
            if not cls_dir.exists():
                print(f"  ⚠️ Missing: {cls_dir}")
                continue
            label = self.class_to_idx[cls]
            for p in cls_dir.rglob('*'):
                if p.suffix.lower() in self.VALID_EXT:
                    self.samples.append((str(p), label))
        print(f"  Loaded {len(self.samples)} images from {root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

    def get_class_weights(self):
        counts = np.zeros(len(self.class_names))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_names)
        return torch.FloatTensor(weights)

    def get_class_distribution(self):
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return dict(counts)

# %%
train_ds = ChestXrayDataset(f"{CFG['data_dir']}/train", CFG['class_names'], train_tf)
val_ds   = ChestXrayDataset(f"{CFG['data_dir']}/val",   CFG['class_names'], val_tf)
test_ds  = ChestXrayDataset(f"{CFG['data_dir']}/test",  CFG['class_names'], val_tf)

# num_workers=0 for Colab stability
nw = 0
train_loader = DataLoader(train_ds, CFG['batch_size'], shuffle=True,  num_workers=nw, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   CFG['batch_size'], shuffle=False, num_workers=nw, pin_memory=True)
test_loader  = DataLoader(test_ds,  CFG['batch_size'], shuffle=False, num_workers=nw, pin_memory=True)

# Print class distribution
print("\nClass Distribution:")
for split_name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
    dist = ds.get_class_distribution()
    print(f"  {split_name}: {dist}")

# %% [markdown]
# ## 5. Model Architecture

# %%
class CustomCNN(nn.Module):
    """4-block custom CNN for binary medical image classification."""
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        def conv_block(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(oc), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(conv_block(3,32), conv_block(32,64),
                                      conv_block(64,128), conv_block(128,256))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256,512),
                                        nn.ReLU(True), nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x): return self.classifier(self.pool(self.features(x)))
    def get_gradcam_layer(self): return self.features[-1]

def get_model(cfg):
    name = cfg['model_name']
    nc   = cfg['num_classes']
    pt   = cfg.get('pretrained', True)
    fb   = cfg.get('freeze_backbone', False)

    if name == 'resnet50':
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pt else None
        m = models.resnet50(weights=w)
        if fb:
            for p in m.parameters(): p.requires_grad = False
        inf = m.fc.in_features
        m.fc = nn.Sequential(nn.Linear(inf, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, nc))
        m.get_gradcam_layer = lambda: m.layer4

    elif name == 'densenet121':
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pt else None
        m = models.densenet121(weights=w)
        if fb:
            for p in m.parameters(): p.requires_grad = False
        inf = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Linear(inf, 512), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(512, nc))
        m.get_gradcam_layer = lambda: m.features.denseblock4

    elif name == 'efficientnet_b0':
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pt else None
        m = models.efficientnet_b0(weights=w)
        if fb:
            for p in m.parameters(): p.requires_grad = False
        inf = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(0.4, inplace=True), nn.Linear(inf, nc))
        m.get_gradcam_layer = lambda: m.features[-1]

    elif name == 'custom_cnn':
        m = CustomCNN(num_classes=nc)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: resnet50, densenet121, efficientnet_b0, custom_cnn")

    total   = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Model: {name} | Total: {total:,} | Trainable: {train_p:,}")
    return m

model = get_model(CFG).to(device)

# %% [markdown]
# ## 6. Training Loop

# %%
def train_model(model, train_loader, val_loader, cfg, device):
    epochs   = cfg['epochs']
    use_amp  = device.type == 'cuda'

    # Class-weighted loss for imbalanced data
    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # AMP scaler — PyTorch 2.x compatible
    try:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda') if use_amp else None
    except ImportError:
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {'tl':[], 'ta':[], 'vl':[], 'va':[]}
    best_acc, patience_ct = 0.0, 0
    t_start = time.time()

    for epoch in range(1, epochs+1):
        t0 = time.time()
        # ---- Train ----
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train", leave=False):
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out  = model(imgs); loss = criterion(out, lbls)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == lbls).sum().item()
            tt += imgs.size(0)
        tl /= tt; ta = tc/tt

        # ---- Validate ----
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val", leave=False):
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(imgs); loss = criterion(out, lbls)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == lbls).sum().item()
                vt += imgs.size(0)
        vl /= vt; va = vc/vt

        scheduler.step()
        history['tl'].append(tl); history['ta'].append(ta)
        history['vl'].append(vl); history['va'].append(va)
        lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:02d}/{epochs} | T {tl:.4f}/{ta:.4f} | V {vl:.4f}/{va:.4f} | LR={lr:.2e} | {time.time()-t0:.1f}s")

        if va > best_acc:
            best_acc = va
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': va, 'val_loss': vl}, cfg['model_path'])
            print(f"    ↳ Best model saved (val_acc={va:.4f})")
            patience_ct = 0
        else:
            patience_ct += 1
            if patience_ct >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch}"); break

    # Load best weights
    ck = torch.load(cfg['model_path'], map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best val_acc: {best_acc:.4f}")
    print(f"{'='*50}")
    return history

# %%
history = train_model(model, train_loader, val_loader, CFG, device)

# %% [markdown]
# ## 7. Training Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=16, fontweight='bold')
eps = range(1, len(history['tl'])+1)

axes[0].plot(eps, history['tl'], 'b-o', ms=3, label='Train')
axes[0].plot(eps, history['vl'], 'r-o', ms=3, label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(eps, history['ta'], 'b-o', ms=3, label='Train')
axes[1].plot(eps, history['va'], 'r-o', ms=3, label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].set_title('Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plots/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Evaluation

# %%
@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for imgs, lbls in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, 1)
        all_labels.append(lbls.cpu()); all_probs.append(probs.cpu())
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()
    preds  = probs.argmax(1)
    return labels, preds, probs

labels, preds, probs = get_preds(model, test_loader, device)

# Classification report
report = classification_report(labels, preds, target_names=CFG['class_names'])
print(report)

# Save report
with open(f"{OUT_DIR}/plots/classification_report.txt", 'w') as f:
    f.write(report)

# Compute metrics
metrics = {
    'accuracy':  round(accuracy_score(labels, preds), 4),
    'precision': round(precision_score(labels, preds, average='binary'), 4),
    'recall':    round(recall_score(labels, preds, average='binary'), 4),
    'f1':        round(f1_score(labels, preds, average='binary'), 4),
    'roc_auc':   round(roc_auc_score(labels, probs[:,1]), 4),
}
print("\nMetrics:", metrics)

# Save metrics to JSON
with open(f"{OUT_DIR}/plots/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# %%
# Confusion Matrix
cm = confusion_matrix(labels, preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Confusion Matrix", fontsize=15, fontweight='bold')

cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
for ax, data, fmt, title in zip(axes, [cm, cm_pct], ['d', '.1f'], ['Raw Counts', 'Normalized (%)']):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CFG['class_names'], yticklabels=CFG['class_names'],
                linewidths=0.5, ax=ax)
    ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plots/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# ROC Curve
fpr, tpr, _ = roc_curve(labels, probs[:,1])
auc_val = roc_auc_score(labels, probs[:,1])

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.4f}')
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plots/roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
# Precision-Recall Curve (critical for imbalanced medical data)
prec_vals, rec_vals, _ = precision_recall_curve(labels, probs[:,1])
ap = average_precision_score(labels, probs[:,1])

plt.figure(figsize=(8,6))
plt.plot(rec_vals, prec_vals, lw=2, label=f'AP = {ap:.4f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
plt.legend(loc='lower left'); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/plots/precision_recall_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Grad-CAM & Grad-CAM++ Explanations

# %%
class GradCAM:
    """Standard Grad-CAM — spatial activation heatmap."""
    def __init__(self, model, layer):
        self.model = model; self._act = None; self._grad = None
        self._fh = layer.register_forward_hook(lambda m,i,o: setattr(self,'_act',o.detach()))
        self._bh = layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'_grad',go[0].detach()))

    def __call__(self, x, cls=None):
        self.model.eval()
        out = self.model(x)
        if cls is None: cls = int(out.argmax(1).item())
        self.model.zero_grad()
        score = out[0, cls]
        score.backward()
        w = self._grad.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self._act).sum(1)).squeeze().cpu().numpy()
        cam -= cam.min(); cam /= cam.max() + 1e-8
        return cam, cls

    def remove(self): self._fh.remove(); self._bh.remove()


class GradCAMPlusPlus:
    """Grad-CAM++ — second-order gradients for sharper heatmaps."""
    def __init__(self, model, layer):
        self.model = model; self._act = None; self._grad = None
        self._fh = layer.register_forward_hook(lambda m,i,o: setattr(self,'_act',o.detach()))
        self._bh = layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'_grad',go[0].detach()))

    def __call__(self, x, cls=None):
        self.model.eval()
        out = self.model(x)
        if cls is None: cls = int(out.argmax(1).item())
        self.model.zero_grad()
        score = out[0, cls]
        score.backward()
        grads = self._grad; acts = self._act
        grad_sq = grads ** 2; grad_cb = grads ** 3
        sum_acts_grad_cb = (acts * grad_cb).sum(dim=(2,3), keepdim=True)
        alpha = torch.relu(grads) / (2 * grad_sq + sum_acts_grad_cb + 1e-8)
        weights = (alpha * torch.relu(grads)).sum(dim=(2,3), keepdim=True)
        cam = torch.relu((weights * acts).sum(1)).squeeze().cpu().numpy()
        cam -= cam.min(); cam /= cam.max() + 1e-8
        return cam, cls

    def remove(self): self._fh.remove(); self._bh.remove()


def get_target_layer(model, name):
    if name == 'resnet50':   return model.layer4
    if name == 'densenet121': return model.features.denseblock4
    if name == 'efficientnet_b0': return model.features[-1]
    if hasattr(model, 'get_gradcam_layer'): return model.get_gradcam_layer()
    return list(model.features.children())[-1]

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    t = tensor.clone().cpu()
    if t.dim() == 4: t = t.squeeze(0)
    for ch, m, s in zip(t, mean, std): ch.mul_(s).add_(m)
    t = torch.clamp(t, 0, 1)
    return (t.permute(1,2,0).numpy() * 255).astype(np.uint8)

# %%
def visualize_gradcam(model, loader, cfg, device, n=8, save_dir=None):
    if save_dir is None: save_dir = OUT_DIR + '/gradcam'
    os.makedirs(save_dir, exist_ok=True)
    imgs, lbls = next(iter(loader))
    n = min(n, imgs.size(0))
    layer = get_target_layer(model, cfg['model_name'])

    cam_std = GradCAM(model, layer)
    cam_pp  = GradCAMPlusPlus(model, layer)

    fig, axes = plt.subplots(n, 4, figsize=(20, 4*n))
    if n == 1: axes = axes[None, :]

    for i in range(n):
        x = imgs[i].unsqueeze(0).to(device)
        orig = denormalize(imgs[i])

        hm_std, pred = cam_std(x)
        hm_pp, _     = cam_pp(x, cls=pred)

        conf = torch.softmax(model(x), 1)[0, pred].item()
        true_cls = cfg['class_names'][lbls[i].item()]
        pred_cls = cfg['class_names'][pred]

        # Overlay
        hm_std_resized = cv2.resize(hm_std, (orig.shape[1], orig.shape[0]))
        hm_pp_resized  = cv2.resize(hm_pp,  (orig.shape[1], orig.shape[0]))
        overlay_std = (0.5 * cv2.applyColorMap(np.uint8(255*hm_std_resized), cv2.COLORMAP_JET)[...,::-1] + 0.5 * orig).astype(np.uint8)
        overlay_pp  = (0.5 * cv2.applyColorMap(np.uint8(255*hm_pp_resized),  cv2.COLORMAP_JET)[...,::-1] + 0.5 * orig).astype(np.uint8)

        axes[i,0].imshow(orig); axes[i,0].set_title(f'True: {true_cls}', fontsize=9); axes[i,0].axis('off')
        axes[i,1].imshow(overlay_std); axes[i,1].set_title(f'Grad-CAM: {pred_cls} ({conf:.0%})', fontsize=9); axes[i,1].axis('off')
        axes[i,2].imshow(overlay_pp); axes[i,2].set_title(f'Grad-CAM++', fontsize=9); axes[i,2].axis('off')
        axes[i,3].imshow(hm_pp, cmap='jet'); axes[i,3].set_title('Heatmap++', fontsize=9); axes[i,3].axis('off')

        cv2.imwrite(f'{save_dir}/gradcam_{i:03d}_{pred_cls}.png',
                    cv2.cvtColor(overlay_std, cv2.COLOR_RGB2BGR))

    cam_std.remove(); cam_pp.remove()
    plt.suptitle("Grad-CAM & Grad-CAM++ Explanations", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/gradcam_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Grad-CAM grid saved → {save_dir}/gradcam_grid.png")

visualize_gradcam(model, test_loader, CFG, device, n=8)

# %% [markdown]
# ## 10. SHAP Explanations

# %%
import shap

def visualize_shap(model, loader, cfg, device, n_test=5, n_bg=50):
    model.eval()
    imgs, _ = next(iter(loader))

    bg_imgs = imgs[:n_bg].to(device)
    test_imgs = imgs[:n_test].to(device)

    print("Computing SHAP values (this may take ~2-5 min on GPU)...")
    explainer   = shap.DeepExplainer(model, bg_imgs)
    shap_values = explainer.shap_values(test_imgs)

    # Normalize for newer SHAP versions
    if isinstance(shap_values, np.ndarray):
        shap_values = [shap_values[..., i] for i in range(shap_values.shape[-1])]

    save_dir = OUT_DIR + '/shap'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_test):
        fig, axes = plt.subplots(1, len(cfg['class_names'])+1, figsize=(5*(len(cfg['class_names'])+1), 4))
        fig.patch.set_facecolor('#0d1117')

        orig = denormalize(imgs[i])
        axes[0].imshow(orig); axes[0].set_title('Original', color='white'); axes[0].axis('off')

        for ci, cls_name in enumerate(cfg['class_names']):
            sv = shap_values[ci][i]
            sv = np.transpose(sv, (1,2,0))
            sal = np.abs(sv).mean(-1)
            sal = (sal-sal.min())/(sal.max()-sal.min()+1e-8)
            im = axes[ci+1].imshow(sal, cmap='RdBu_r', vmin=0, vmax=1)
            axes[ci+1].set_title(f'SHAP — {cls_name}', color='white'); axes[ci+1].axis('off')
            plt.colorbar(im, ax=axes[ci+1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_sample_{i:03d}.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Summary bar
    mean_abs = [np.abs(sv).mean() for sv in shap_values]
    plt.figure(figsize=(8,4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(cfg['class_names'])))
    bars = plt.bar(cfg['class_names'], mean_abs, color=colors)
    plt.bar_label(bars, fmt='%.4f', padding=3)
    plt.ylabel('Mean |SHAP|'); plt.title('Mean Absolute SHAP per Class')
    plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_summary_bar.png', dpi=150)
    plt.show()
    print(f"✅ SHAP saved → {save_dir}")

visualize_shap(model, test_loader, CFG, device, n_test=5, n_bg=50)

# %% [markdown]
# ## 11. Single Image Analysis (Upload Test)

# %%
def analyze_single_image(model, img_path, cfg, device):
    """Analyze a single image with prediction + Grad-CAM + Grad-CAM++ + SHAP."""
    pil = Image.open(img_path).convert('RGB')
    x = val_tf(pil).unsqueeze(0).to(device)

    # Prediction
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, 1)[0]
    pred_idx = int(probs_t.argmax().item())
    print(f"\nPrediction: {cfg['class_names'][pred_idx]} ({probs_t[pred_idx]:.2%})")
    for i, cls in enumerate(cfg['class_names']):
        print(f"  {cls}: {probs_t[i]:.2%}")

    orig = np.array(pil.resize((cfg['image_size'], cfg['image_size'])))

    # Grad-CAM + Grad-CAM++
    layer = get_target_layer(model, cfg['model_name'])
    gcam = GradCAM(model, layer)
    gcampp = GradCAMPlusPlus(model, layer)

    hm_std, _ = gcam(x, cls=pred_idx)
    hm_pp, _  = gcampp(x, cls=pred_idx)
    gcam.remove(); gcampp.remove()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(orig); axes[0].set_title('Original'); axes[0].axis('off')

    hm_r = cv2.resize(hm_std, (orig.shape[1], orig.shape[0]))
    overlay = (0.5 * cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET)[...,::-1] + 0.5 * orig).astype(np.uint8)
    axes[1].imshow(overlay); axes[1].set_title('Grad-CAM'); axes[1].axis('off')

    hm_r2 = cv2.resize(hm_pp, (orig.shape[1], orig.shape[0]))
    overlay2 = (0.5 * cv2.applyColorMap(np.uint8(255*hm_r2), cv2.COLORMAP_JET)[...,::-1] + 0.5 * orig).astype(np.uint8)
    axes[2].imshow(overlay2); axes[2].set_title('Grad-CAM++'); axes[2].axis('off')

    # SHAP
    bg_imgs, _ = next(iter(test_loader))
    bg = bg_imgs[:20].to(device)
    explainer = shap.DeepExplainer(model, bg)
    shap_vals = explainer.shap_values(x)
    if isinstance(shap_vals, np.ndarray):
        shap_vals = [shap_vals[..., i] for i in range(shap_vals.shape[-1])]
    sv  = np.transpose(shap_vals[pred_idx][0], (1,2,0))
    sal = np.abs(sv).mean(-1)
    sal = (sal-sal.min())/(sal.max()-sal.min()+1e-8)
    axes[3].imshow(sal, cmap='RdBu_r'); axes[3].set_title('SHAP'); axes[3].axis('off')

    plt.suptitle(f"Prediction: {cfg['class_names'][pred_idx]} ({probs_t[pred_idx]:.1%})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show()

# Test with a sample
sample = test_ds.samples[0][0]
analyze_single_image(model, sample, CFG, device)

# %% [markdown]
# ## 12. Summary & Next Steps
#
# ### Results Summary
# - **Model**: ResNet-50 (pretrained on ImageNet)
# - **Accuracy**: See metrics above
# - **Recall (Sensitivity)**: Critical metric for medical AI — catches pneumonia cases
# - **Explainability**: Grad-CAM + Grad-CAM++ + SHAP provide complementary views
#
# ### What's New in Enhanced Version
# 1. ✅ **DenseNet-121** architecture support
# 2. ✅ **Grad-CAM++** for sharper, more discriminative heatmaps
# 3. ✅ **Precision-Recall curve** (vital for imbalanced medical data)
# 4. ✅ **Metrics persisted** to JSON + classification report to text
# 5. ✅ **All PyTorch 2.x deprecations** fixed (AMP, torch.load)
# 6. ✅ **SHAP version compatibility** for both old and new API
# 7. ✅ **Gradient clipping** in all training paths (AMP and non-AMP)
# 8. ✅ **tqdm progress bars** for training/evaluation
#
# ### Future Work
# - [ ] Multi-label classification (NIH ChestX-ray14 — 14 pathologies)
# - [ ] Vision Transformer (ViT) with attention visualization
# - [ ] Monte-Carlo Dropout for uncertainty quantification
# - [ ] Docker + FastAPI for production deployment
#
# > ⚠️ **Clinical Disclaimer**: This tool is for research/educational purposes only.
# > Always rely on licensed radiologists for clinical diagnosis.
