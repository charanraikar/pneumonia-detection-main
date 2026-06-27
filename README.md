AI-powered pneumonia detection from chest X-rays using ConvNeXt-Base + DDPM synthesis + Grad-CAM explainability. 95.19% accuracy | AUC 0.987 | FastAPI web app.

# 🫁 PneumoScan — AI-Powered Pneumonia Detection

An end-to-end medical imaging pipeline that detects pneumonia from chest X-rays
using ConvNeXt-Base, explains predictions with Grad-CAM heatmaps, synthesises
new X-rays using a DDPM, and serves everything through a FastAPI web application.

---

## 🏆 Results

| Metric         | Score   |
|----------------|---------|
| Test Accuracy  | 95.19%  |
| AUC-ROC        | 0.987   |
| Pneumonia F1   | 0.9625  |
| Normal F1      | 0.9330  |
| Test Samples   | 624     |

---

## 🧠 Architecture Overview

Chest X-ray Input

│

▼

Medical Preprocessing (Bilateral Filter + CLAHE)

│

▼

ConvNeXt-Base (ImageNet-22k pretrained)

└── Global Average Pooling [1024]

└── LayerNorm → Dropout(0.4)

└── Linear(1024→512) → GELU

└── Dropout(0.3) → Linear(512→2)

│

▼

┌─────────────┬──────────────┬─────────────────┐

│  Prediction │   Grad-CAM   │   Risk Scorer   │

│ NORMAL /    │  Heatmap     │ LOW / MODERATE  │

│ PNEUMONIA   │  Overlay     │ HIGH / CRITICAL │

└─────────────┴──────────────┴─────────────────┘

---

## 🗂️ Project Structure

pneumonia_project/

│

├── main.py              # Entry point — CLI with --mode all/train/gradcam/ddpm/infer

├── config.py            # All hyperparameters and paths

├── model.py             # ConvNeXt-Base + custom head + LabelSmoothing loss

├── train.py             # Training loop with AMP, early stopping, LR scheduler

├── dataset.py           # PneumoniaDataset, augmentations, weighted sampler

├── gradcam.py           # Grad-CAM engine + RiskScorer + visualisation

├── ddpm.py              # DDPM UNet + cosine noise schedule + X-ray synthesis

├── app.py               # FastAPI backend — /predict, /health endpoints

│

├── templates/

│   └── index.html       # Web UI frontend

│

├── checkpoints/         # best_model.pth saved here

├── results/             # Grad-CAM output images

├── ddpm_outputs/        # Generated X-rays + diffusion process

├── logs/                # TensorBoard logs

│

├── requirements.txt     # ML dependencies

├── requirements_web.txt # FastAPI dependencies

├── setup_env.py         # Environment checker

├── run_all.bat          # One-click full pipeline (Windows)

└── start_webapp.bat     # Launch web app (Windows)

---

## ⚙️ Pipeline Stages

### Stage 1 — DDPM X-ray Synthesis
- UNet trained from scratch on grayscale chest X-rays (64×64)
- Cosine noise schedule over 1000 timesteps
- Generates synthetic X-rays to augment the minority class
- Visualises the full forward noising process (t=0 → t=999)

### Stage 2 — ConvNeXt-Base Classification
- Backbone: `convnext_base` via `timm` (ImageNet-22k pretrained)
- Custom head with LayerNorm, GELU, dual Dropout
- Label smoothing (ε=0.1) + class-weighted loss
- Differential LR: backbone 5e-5, head 2e-4
- Mixed precision (AMP) + gradient clipping
- Stratified 85/15 train/val split (fixes Kaggle's 16-image val set)
- WeightedRandomSampler to handle class imbalance

### Stage 3 — Grad-CAM + Risk Scoring
- Hooks into `backbone.stages[-1]` for activation maps
- Risk Score = 0.7 × P(Pneumonia) + 0.3 × CAM intensity (top 20% pixels)
- Four risk levels: LOW / MODERATE / HIGH / CRITICAL
- 4-panel output: Original | Overlay | Heatmap | Risk Dashboard

### Stage 4 — FastAPI Web App
- Upload any chest X-ray → instant prediction
- Returns: prediction, confidence, risk score, Grad-CAM overlay, risk dashboard
- All images returned as base64 PNG for the frontend

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/charanraikar/pneumoscan.git
cd pneumoscan
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python setup_env.py          # Verify everything is ready
```

### 2. Download Dataset
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
mkdir data && unzip chest-xray-pneumonia.zip -d data/
```

### 3. Run Full Pipeline
```bash
python main.py --mode all
# or step by step:
python main.py --mode ddpm
python main.py --mode train
python main.py --mode gradcam
```

### 4. Single Image Inference
```bash
python main.py --mode infer --image path/to/xray.jpeg
```

### 5. Launch Web App
```bash
pip install -r requirements_web.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## 📊 Training Details

| Parameter          | Value                     |
|--------------------|---------------------------|
| Architecture       | ConvNeXt-Base             |
| Pretrained         | ImageNet-22k              |
| Image Size         | 224 × 224                 |
| Batch Size         | 32                        |
| Optimizer          | AdamW                     |
| Backbone LR        | 5e-5                      |
| Head LR            | 2e-4                      |
| Weight Decay       | 5e-4                      |
| Drop Path Rate     | 0.4                       |
| Label Smoothing    | 0.1                       |
| Mixed Precision    | ✅ AMP                    |
| Early Stopping     | 8 epochs patience         |
| Max Epochs         | 50                        |
| Augmentations      | Albumentations pipeline   |

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, timm, ConvNeXt
- **Diffusion:** Custom DDPM UNet
- **Explainability:** Grad-CAM
- **Augmentation:** Albumentations
- **Web Backend:** FastAPI, Uvicorn
- **Visualisation:** Matplotlib, Seaborn, OpenCV
- **Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 📁 Dataset

The Kaggle chest X-ray dataset contains:
- **Train:** ~5,216 images (NORMAL + PNEUMONIA)
- **Val:** 16 images (merged into train; proper 85/15 stratified split applied)
- **Test:** 624 images

---

## 👤 Author

**Charan Arunkumar Raikar**
M.Tech Computer Science — Manipal Institute of Technology, Bengaluru
[LinkedIn](https://www.linkedin.com/in/charanraikar) • [GitHub](https://github.com/charanraikar)

pneumonia-detection  chest-xray  convnext  deep-learning  grad-cam
ddpm  diffusion-model  medical-imaging  fastapi  pytorch
explainable-ai  transfer-learning  image-classification  python
