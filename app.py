# app.py - FastAPI backend for Pneumonia Detection Web App
import io
import os
import sys
import base64
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Add project root to path so we can import our modules ────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, GRADCAM_CONFIG, RISK_CONFIG, CHECKPOINT_DIR
from model import ConvNeXtPneumonia
from dataset import PneumoniaDataset, get_val_transforms
from gradcam import GradCAM, RiskScorer, overlay_heatmap, tensor_to_rgb

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Pneumonia Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────────────────────
device     = None
model      = None
cam_engine = None

@app.on_event("startup")
async def load_model():
    global device, model, cam_engine

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: No checkpoint found at {ckpt_path}")
        return

    model = ConvNeXtPneumonia()
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    cam_engine = GradCAM(model, model.get_grad_cam_target_layer())
    print(f"  Model loaded. Val AUC: {ckpt.get('val_auc', 'N/A')}")

# ─────────────────────────────────────────────────────────────
# HELPER: image → base64 PNG string
# ─────────────────────────────────────────────────────────────
def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def arr_to_b64(arr: np.ndarray) -> str:
    """numpy RGB uint8 → base64 PNG"""
    img_pil = Image.fromarray(arr)
    buf     = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────────────────────────
# HELPER: build risk dashboard figure
# ─────────────────────────────────────────────────────────────
def build_risk_figure(risk_info: dict) -> str:
    fig, ax = plt.subplots(figsize=(4, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    color = risk_info["color"]
    score = risk_info["risk_score"]

    # Gauge bar
    ax.barh(8.5, 8, height=0.7, left=1, color="#1c2333", zorder=1)
    ax.barh(8.5, 8 * score / 100, height=0.7, left=1, color=color, zorder=2)
    ax.text(5, 9.4, f"Risk Score: {score}%",
            ha="center", va="center", color="white",
            fontsize=13, fontweight="bold")

    # Level badge
    ax.text(5, 7.2, risk_info["risk_level"],
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1c1c2e",
                      edgecolor=color, linewidth=2.5))

    # Metrics
    metrics = [
        ("Pneumonia Prob", f"{risk_info['pneumonia_prob']}%"),
        ("CAM Intensity",  f"{risk_info['cam_intensity']}%"),
        ("Prediction",     "PNEUMONIA" if risk_info["pneumonia_prob"] >= 50 else "NORMAL"),
    ]
    for i, (label, val) in enumerate(metrics):
        y = 5.5 - i * 1.3
        ax.text(1.2, y, f"{label}:", color="#8b949e", fontsize=9, va="center")
        ax.text(8.8, y, val, color="white", fontsize=9, va="center",
                ha="right", fontweight="bold")

    # Recommendation
    rec = risk_info["recommendation"]
    words = rec.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= 34:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)

    ax.text(5, 1.6, "\n".join(lines),
            ha="center", va="center", color="#c9d1d9",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                      edgecolor="#30363d"))

    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64

# ─────────────────────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Check checkpoints/best_model.pth")

    # ── Read & validate image ─────────────────────────────────
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 20MB)")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_np = np.array(pil_img)

    # ── Preprocess ────────────────────────────────────────────
    img_enhanced = PneumoniaDataset._enhance_xray(img_np)
    transform    = get_val_transforms()
    img_tensor   = transform(image=img_enhanced)["image"].to(device)

    # ── Grad-CAM ──────────────────────────────────────────────
    heatmap, pred_class, confidence = cam_engine(img_tensor, class_idx=1)

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pneumonia_prob = float(probs[1])
    risk_info      = RiskScorer.compute(pneumonia_prob, heatmap)

    # ── Build output images ───────────────────────────────────
    # 1. Original (resized to 224x224 for consistency)
    orig_rgb  = tensor_to_rgb(img_tensor)
    orig_b64  = arr_to_b64(orig_rgb)

    # 2. Grad-CAM overlay
    overlay   = overlay_heatmap(orig_rgb, heatmap)
    overlay_b64 = arr_to_b64(overlay)

    # 3. Activation heatmap
    hm_fig, hm_ax = plt.subplots(figsize=(3, 3), facecolor="#0d1117")
    hm_ax.set_facecolor("#0d1117")
    im = hm_ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=hm_ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("Activation", color="white", fontsize=8)
    hm_ax.axis("off")
    plt.tight_layout()
    heatmap_b64 = fig_to_b64(hm_fig)
    plt.close(hm_fig)

    # 4. Risk dashboard
    risk_b64 = build_risk_figure(risk_info)

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    return JSONResponse({
        "prediction"     : CLASS_NAMES[pred_class],
        "confidence"     : round(confidence * 100, 1),
        "pneumonia_prob" : round(pneumonia_prob * 100, 1),
        "normal_prob"    : round(float(probs[0]) * 100, 1),
        "risk_score"     : risk_info["risk_score"],
        "risk_level"     : risk_info["risk_level"],
        "risk_color"     : risk_info["color"],
        "recommendation" : risk_info["recommendation"],
        "images": {
            "original"   : orig_b64,
            "overlay"    : overlay_b64,
            "heatmap"    : heatmap_b64,
            "risk"       : risk_b64,
        }
    })

# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "device"      : str(device),
    }

# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()