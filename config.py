# config.py - Central configuration for Pneumonia Detection Project
import torch
import os

# ─────────────────────────────────────────────────────────────
# GPU CONFIGURATION — Optimized for NVIDIA RTX A2000 12GB VRAM
# ─────────────────────────────────────────────────────────────
def setup_device():
    """Configure GPU device with memory optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found. Running on CPU.")
    return device

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "chest_xray")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VAL_DIR         = os.path.join(DATA_DIR, "val")
TEST_DIR        = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
DDPM_DIR        = os.path.join(BASE_DIR, "ddpm_outputs")

for d in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR, DDPM_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "architecture"      : "convnext_base",
    "num_classes"       : 2,
    "pretrained"        : True,
    "drop_path_rate"    : 0.4,     # increased from 0.2 to reduce overfitting
    "image_size"        : 224,
}

# ─────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs"            : 50,
    "batch_size"        : 32,
    "learning_rate"     : 2e-4,
    "weight_decay"      : 5e-4,    # increased from 1e-4
    "warmup_epochs"     : 5,       # increased from 3
    "num_workers"       : 4,       # reduced from 8 — fixes Windows multiprocessing
    "pin_memory"        : True,
    "gradient_clip"     : 1.0,
    "accumulation_steps": 1,
    "mixed_precision"   : True,
    "early_stopping"    : 8,       # increased from 5
    "seed"              : 42,
    "classes"           : ["NORMAL", "PNEUMONIA"],
    "val_split"         : 0.15,    # 15% of train+val merged for proper validation
}

# ─────────────────────────────────────────────────────────────
# DDPM CONFIGURATION
# ─────────────────────────────────────────────────────────────
DDPM_CONFIG = {
    "timesteps"         : 1000,
    "beta_start"        : 1e-4,
    "beta_end"          : 0.02,
    "image_size"        : 64,
    "channels"          : 1,
    "batch_size"        : 8,
    "epochs"            : 50,
    "lr"                : 1e-4,
    "schedule"          : "cosine",
    "num_generate"      : 16,
}

# ─────────────────────────────────────────────────────────────
# GRAD-CAM CONFIGURATION
# ─────────────────────────────────────────────────────────────
GRADCAM_CONFIG = {
    "target_layer"      : "stages.3",
    "colormap"          : "jet",
    "alpha"             : 0.5,
    "resize_to"         : (224, 224),
}

# ─────────────────────────────────────────────────────────────
# RISK SCORE THRESHOLDS
# ─────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "LOW"       : (0.0,  0.35),
    "MODERATE"  : (0.35, 0.65),
    "HIGH"      : (0.65, 0.85),
    "CRITICAL"  : (0.85, 1.01),
    "colors"    : {
        "LOW"      : "#2ecc71",
        "MODERATE" : "#f39c12",
        "HIGH"     : "#e67e22",
        "CRITICAL" : "#e74c3c",
    }
}