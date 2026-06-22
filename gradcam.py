# gradcam.py - Gradient-weighted Class Activation Mapping + Risk Scoring
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional
from config import GRADCAM_CONFIG, RISK_CONFIG

# ─────────────────────────────────────────────────────────────
# GRAD-CAM ENGINE
# ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into any target layer and computes activation maps.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def __call__(self, img_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Returns:
            heatmap    : np.ndarray (H, W) in [0, 1]
            pred_class : predicted class index
            confidence : softmax confidence for predicted class
        """
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
        img_tensor = img_tensor.requires_grad_(True)

        # Forward
        logits     = self.model(img_tensor)
        probs      = F.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if class_idx is None:
            class_idx = pred_class

        # Backward for target class
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Compute Grad-CAM
        grads = self.gradients[0]          # (C, H, W)
        acts  = self.activations[0]        # (C, H, W)
        weights = grads.mean(dim=(1, 2))   # Global average pool
        cam   = (weights[:, None, None] * acts).sum(0)
        cam   = F.relu(cam).cpu().numpy()

        # Normalise
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        heatmap = cv2.resize(cam, GRADCAM_CONFIG["resize_to"])
        return heatmap, pred_class, confidence

    def get_all_class_cams(self, img_tensor: torch.Tensor):
        """Return CAMs for both classes."""
        cams = {}
        img_tensor = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
        logits = self.model(img_tensor)
        probs  = F.softmax(logits, dim=1)
        for c in range(logits.shape[1]):
            self.model.zero_grad()
            logits[0, c].backward(retain_graph=True)
            grads = self.gradients[0]
            acts  = self.activations[0]
            w     = grads.mean(dim=(1, 2))
            cam   = (w[:, None, None] * acts).sum(0)
            cam   = F.relu(cam).cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            cams[c] = cv2.resize(cam, GRADCAM_CONFIG["resize_to"])
        return cams, probs.detach().cpu().numpy()[0]

# ─────────────────────────────────────────────────────────────
# RISK SCORE CALCULATOR
# ─────────────────────────────────────────────────────────────
class RiskScorer:
    """
    Combines model confidence with CAM intensity to produce
    a clinically-interpretable risk score.

    Risk Score = α * P(PNEUMONIA) + β * CAM_intensity_ratio
    where α=0.7, β=0.3
    """
    ALPHA = 0.70   # Weight for model probability
    BETA  = 0.30   # Weight for CAM intensity signal

    @staticmethod
    def compute(pneumonia_prob: float, heatmap: np.ndarray) -> dict:
        # CAM intensity: mean of top-20% activated pixels (lung region focus)
        thresh       = np.percentile(heatmap, 80)
        intensity    = heatmap[heatmap >= thresh].mean() if (heatmap >= thresh).any() else 0.0
        risk_score   = RiskScorer.ALPHA * pneumonia_prob + RiskScorer.BETA * float(intensity)
        risk_score   = float(np.clip(risk_score, 0.0, 1.0))
        level        = RiskScorer._classify(risk_score)
        return {
            "risk_score"      : round(risk_score * 100, 1),   # Percentage
            "risk_level"      : level,
            "pneumonia_prob"  : round(pneumonia_prob * 100, 1),
            "cam_intensity"   : round(float(intensity) * 100, 1),
            "color"           : RISK_CONFIG["colors"][level],
            "recommendation"  : RiskScorer._recommendation(level),
        }

    @staticmethod
    def _classify(score: float) -> str:
        for level, (lo, hi) in RISK_CONFIG.items():
            if level == "colors":
                continue
            if lo <= score < hi:
                return level
        return "CRITICAL"

    @staticmethod
    def _recommendation(level: str) -> str:
        recs = {
            "LOW"      : "No pneumonia detected. Routine follow-up recommended.",
            "MODERATE" : "Possible early pneumonia. Clinical correlation advised.",
            "HIGH"     : "Strong pneumonia indicators. Immediate clinical review required.",
            "CRITICAL" : "Critical pneumonia risk. Urgent intervention recommended.",
        }
        return recs.get(level, "")

# ─────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize ImageNet-normalised tensor → RGB uint8."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = GRADCAM_CONFIG["alpha"]) -> np.ndarray:
    """Blend Grad-CAM heatmap over X-ray image."""
    heatmap_u8  = (heatmap * 255).astype(np.uint8)
    colormap    = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colormap    = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    blended     = cv2.addWeighted(img_rgb, 1 - alpha, colormap, alpha, 0)
    return blended

def create_result_figure(img_tensor: torch.Tensor, heatmap: np.ndarray,
                          risk_info: dict, true_label: Optional[int] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a 4-panel clinical result figure:
      [Original] [Heatmap Overlay] [Heatmap Only] [Risk Dashboard]
    """
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    img_rgb = tensor_to_rgb(img_tensor)
    overlay = overlay_heatmap(img_rgb, heatmap)

    fig = plt.figure(figsize=(18, 5), facecolor="#0d1117")
    gs  = GridSpec(1, 4, figure=fig, wspace=0.05)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    panel_titles = ["Original X-ray", "Grad-CAM Overlay",
                    "Activation Map", "Risk Assessment"]
    for ax, title in zip(axs, panel_titles):
        ax.set_facecolor("#0d1117")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel 1: Original
    axs[0].imshow(img_rgb)
    if true_label is not None:
        color  = "#e74c3c" if true_label == 1 else "#2ecc71"
        axs[0].set_xlabel(f"True: {CLASS_NAMES[true_label]}",
                          color=color, fontsize=10, fontweight="bold")

    # Panel 2: Overlay
    axs[1].imshow(overlay)

    # Panel 3: Heatmap only
    im = axs[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Activation", color="white", fontsize=9)

    # Panel 4: Risk Dashboard
    ax = axs[3]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    level_color = risk_info["color"]

    # Risk gauge bar
    score_norm = risk_info["risk_score"] / 100
    bar_w = 8
    ax.barh(8.0, bar_w, height=0.6, left=1, color="#30363d", zorder=1)
    ax.barh(8.0, bar_w * score_norm, height=0.6, left=1, color=level_color, zorder=2)
    ax.text(5, 8.7, f"Risk Score: {risk_info['risk_score']}%",
            ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    # Risk level badge
    ax.text(5, 7.0, risk_info["risk_level"],
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=level_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1c1c2e",
                      edgecolor=level_color, linewidth=2))

    # Metrics
    metrics = [
        ("Pneumonia Prob", f"{risk_info['pneumonia_prob']}%"),
        ("CAM Intensity",  f"{risk_info['cam_intensity']}%"),
        ("Prediction",     CLASS_NAMES[int(risk_info["pneumonia_prob"] >= 50)]),
    ]
    for i, (label, val) in enumerate(metrics):
        y = 5.5 - i * 1.2
        ax.text(1.5, y, f"{label}:", color="#8b949e", fontsize=9, va="center")
        ax.text(8.5, y, val, color="white", fontsize=9, va="center", ha="right",
                fontweight="bold")

    # Recommendation
    rec_lines = _wrap_text(risk_info["recommendation"], 32)
    rec_text  = "\n".join(rec_lines)
    ax.text(5, 1.8, rec_text, ha="center", va="center", color="#c9d1d9",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                      edgecolor="#30363d"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
    return fig

def _wrap_text(text: str, max_len: int):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# ─────────────────────────────────────────────────────────────
# BATCH INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────
def run_gradcam_analysis(model, test_loader, device,
                          save_dir: str, max_images: int = 30):
    import os
    os.makedirs(save_dir, exist_ok=True)
    cam_engine  = GradCAM(model, model.get_grad_cam_target_layer())
    scorer      = RiskScorer()
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    model.eval()
    results  = []
    count    = 0
    n_normal = 0
    n_pneumo = 0
    per_class = max_images // 2   # 15 NORMAL + 15 PNEUMONIA

    # Collect all test samples sorted so PNEUMONIA comes first
    all_batches = list(test_loader)
    # Sort: pneumonia (label=1) first
    all_batches.sort(key=lambda b: b[1][0].item(), reverse=True)

    for imgs, labels, paths in all_batches:
        if count >= max_images:
            break

        true_lbl = labels[0].item()

        # Skip if we already have enough of this class
        if true_lbl == 0 and n_normal >= per_class:
            continue
        if true_lbl == 1 and n_pneumo >= per_class:
            continue

        img_t = imgs[0].to(device)

        heatmap, pred_class, confidence = cam_engine(img_t, class_idx=1)

        with torch.no_grad():
            logits = model(img_t.unsqueeze(0))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pneumonia_prob = float(probs[1])
        risk_info      = scorer.compute(pneumonia_prob, heatmap)

        fname = f"result_{count:04d}_true{CLASS_NAMES[true_lbl]}_pred{CLASS_NAMES[pred_class]}.png"
        fpath = os.path.join(save_dir, fname)
        create_result_figure(img_t, heatmap, risk_info,
                              true_label=true_lbl, save_path=fpath)
        plt.close("all")

        if true_lbl == 0:
            n_normal += 1
        else:
            n_pneumo += 1

        results.append({
            "path"        : paths[0],
            "true_label"  : CLASS_NAMES[true_lbl],
            "pred_label"  : CLASS_NAMES[pred_class],
            "confidence"  : round(confidence * 100, 2),
            **risk_info,
        })
        count += 1
        print(f"  [{count:3d}] {CLASS_NAMES[true_lbl]:9s} -> {CLASS_NAMES[pred_class]:9s} "
              f"| Risk: {risk_info['risk_score']:5.1f}% ({risk_info['risk_level']})")

    cam_engine.remove_hooks()
    print(f"\n  Summary: {n_normal} NORMAL + {n_pneumo} PNEUMONIA images processed")
    return results
