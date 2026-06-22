# main.py - Entry point
# IMPORTANT: multiprocessing fix for Windows must be at the very top
import multiprocessing
multiprocessing.freeze_support()

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from config import (setup_device, DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
                    CHECKPOINT_DIR, RESULTS_DIR, DDPM_DIR)
from dataset import build_dataloaders, PneumoniaDataset, get_val_transforms
from model import ConvNeXtPneumonia, build_model
from train import train, evaluate_test
from ddpm import train_ddpm, visualise_diffusion_process, DDPM_CONFIG
from gradcam import run_gradcam_analysis

# ─────────────────────────────────────────────────────────────
# SINGLE-IMAGE INFERENCE
# ─────────────────────────────────────────────────────────────
def infer_single(image_path: str, device, checkpoint_path: str = None):
    import cv2
    import numpy as np
    from gradcam import GradCAM, RiskScorer, create_result_figure
    import matplotlib.pyplot as plt

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"\n🔍 Inference on: {image_path}")

    model = ConvNeXtPneumonia()
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img   = PneumoniaDataset._enhance_xray(img)
    transform = get_val_transforms()
    img_t = transform(image=img)["image"].to(device)

    cam_engine = GradCAM(model, model.get_grad_cam_target_layer())
    heatmap, pred_class, confidence = cam_engine(img_t, class_idx=1)

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    risk_info   = RiskScorer.compute(float(probs[1]), heatmap)
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    print(f"\n{'='*50}")
    print(f"  🩺 PNEUMONIA DETECTION RESULT")
    print(f"{'='*50}")
    print(f"  Prediction    : {CLASS_NAMES[pred_class]}")
    print(f"  Confidence    : {confidence*100:.1f}%")
    print(f"  Risk Score    : {risk_info['risk_score']}%")
    print(f"  Risk Level    : {risk_info['risk_level']}")
    print(f"  Recommendation: {risk_info['recommendation']}")
    print(f"{'='*50}\n")

    save_path = os.path.join(RESULTS_DIR, "single_inference_result.png")
    fig = create_result_figure(img_t, heatmap, risk_info, save_path=save_path)
    plt.show()
    print(f"  📷 Result saved: {save_path}")
    cam_engine.remove_hooks()

# ─────────────────────────────────────────────────────────────
# GRAD-CAM ONLY
# ─────────────────────────────────────────────────────────────
def run_gradcam_only(device):
    _, _, test_loader, _ = build_dataloaders(TRAIN_DIR, VAL_DIR, TEST_DIR)
    model = ConvNeXtPneumonia()
    ckpt  = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"),
                       map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    gradcam_dir = os.path.join(RESULTS_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    results = run_gradcam_analysis(model, test_loader, device,
                                    save_dir=gradcam_dir, max_images=30)
    print(f"\n✅ Grad-CAM complete. {len(results)} images saved to {gradcam_dir}/")

# ─────────────────────────────────────────────────────────────
# DDPM ONLY
# ─────────────────────────────────────────────────────────────
def run_ddpm_only(device):
    train_loader, _, test_loader, _ = build_dataloaders(TRAIN_DIR, VAL_DIR, TEST_DIR)
    unet, ddpm = train_ddpm(train_loader, device)
    for imgs, _, _ in test_loader:
        visualise_diffusion_process(
            imgs[0], ddpm,
            save_path=os.path.join(DDPM_DIR, "diffusion_process.png")
        )
        break

# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────
def run_full_pipeline(device):
    print("\n" + "="*65)
    print("  🫁  PNEUMONIA DETECTION — FULL PIPELINE")
    print("="*65)

    # Step 1: DDPM
    print("\n📌 STEP 1: DDPM X-ray Synthesis")
    train_loader, _, test_loader_ddpm, _ = build_dataloaders(
        TRAIN_DIR, VAL_DIR, TEST_DIR)
    unet, ddpm = train_ddpm(train_loader, device)
    for imgs, _, _ in test_loader_ddpm:
        visualise_diffusion_process(
            imgs[0], ddpm,
            save_path=os.path.join(DDPM_DIR, "diffusion_process.png")
        )
        break
    del unet, ddpm, train_loader, test_loader_ddpm
    torch.cuda.empty_cache()

    # Step 2: ConvNeXt Training
    print("\n📌 STEP 2: ConvNeXt-Base Training")
    model, test_loader = train(TRAIN_DIR, VAL_DIR, TEST_DIR, device)
    torch.cuda.empty_cache()

    # Step 3: Grad-CAM + Risk Scoring
    print("\n📌 STEP 3: Grad-CAM + Risk Scoring")
    gradcam_dir = os.path.join(RESULTS_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    results = run_gradcam_analysis(model, test_loader, device,
                                    save_dir=gradcam_dir, max_images=30)

    print("\n" + "="*65)
    print("  ✅  PIPELINE COMPLETE")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"  DDPM    → {DDPM_DIR}/")
    print("="*65 + "\n")

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Pipeline")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "train", "gradcam", "ddpm", "infer"])
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = setup_device()

    if not os.path.exists(TRAIN_DIR) and args.mode != "infer":
        print(f"\n❌ Dataset not found at {DATA_DIR}")
        print("   Download: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
        sys.exit(1)

    if args.mode == "all":
        run_full_pipeline(device)
    elif args.mode == "train":
        train(TRAIN_DIR, VAL_DIR, TEST_DIR, device)
    elif args.mode == "gradcam":
        run_gradcam_only(device)
    elif args.mode == "ddpm":
        run_ddpm_only(device)
    elif args.mode == "infer":
        if args.image is None:
            print("❌ --image required for infer mode")
            sys.exit(1)
        infer_single(args.image, device, args.checkpoint)


if __name__ == "__main__":
    main()