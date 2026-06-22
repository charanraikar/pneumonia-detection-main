#!/usr/bin/env python3
# setup_env.py - One-click environment verifier and setup helper
"""
Run this FIRST before anything else:
    python setup_env.py
"""
import subprocess
import sys
import os

def run(cmd, check=True):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ FAILED: {cmd}")
        print(result.stderr)
    return result

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {name} ({vram:.1f} GB VRAM)")
            print(f"  ✅ CUDA: {torch.version.cuda}")
            print(f"  ✅ PyTorch: {torch.__version__}")
            return True
        else:
            print("  ⚠️  No CUDA GPU detected. Check NVIDIA drivers.")
            return False
    except ImportError:
        print("  ❌ PyTorch not installed.")
        return False

def check_packages():
    packages = [
        "torch", "torchvision", "timm", "albumentations",
        "cv2", "sklearn", "matplotlib", "seaborn",
        "tqdm", "diffusers", "einops"
    ]
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — MISSING")
            missing.append(pkg)
    return missing

def check_data():
    data_dirs = [
        "data/chest_xray/train/NORMAL",
        "data/chest_xray/train/PNEUMONIA",
        "data/chest_xray/val/NORMAL",
        "data/chest_xray/val/PNEUMONIA",
        "data/chest_xray/test/NORMAL",
        "data/chest_xray/test/PNEUMONIA",
    ]
    all_ok = True
    for d in data_dirs:
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png"))])
            print(f"  ✅ {d} ({count} images)")
        else:
            print(f"  ❌ {d} — NOT FOUND")
            all_ok = False
    return all_ok

def main():
    print("\n" + "="*55)
    print("  🔧 Pneumonia Project — Environment Check")
    print("="*55)

    print("\n📦 GPU / PyTorch:")
    gpu_ok = check_gpu()

    print("\n📦 Python packages:")
    missing = check_packages()

    print("\n📂 Dataset directories:")
    data_ok = check_data()

    print("\n" + "="*55)
    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("   Install with:")
        print("   pip install " + " ".join(missing))
    if not data_ok:
        print("\n⚠️  Dataset missing! Download steps:")
        print("   1. Install Kaggle CLI: pip install kaggle")
        print("   2. Place kaggle.json in C:/Users/<USER>/.kaggle/")
        print("   3. Run:")
        print("      kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
        print("      mkdir -p data && unzip chest-xray-pneumonia.zip -d data/")
        print("   OR download manually from:")
        print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    if gpu_ok and not missing and data_ok:
        print("\n✅ Everything is ready! Run:")
        print("   python main.py --mode all")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()