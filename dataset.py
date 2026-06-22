# dataset.py - Data loading, preprocessing, and augmentation pipeline
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
from config import TRAIN_CONFIG, MODEL_CONFIG

# ─────────────────────────────────────────────────────────────
# IMAGE NORMALIZATION STATS
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = MODEL_CONFIG["image_size"]

# ─────────────────────────────────────────────────────────────
# ALBUMENTATIONS PIPELINES
# ─────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        # ── Geometry ──────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, p=0.5),
        A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE, p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        # ── Intensity / Quality ──────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
        # ── Regularization ──────────────────────────────────
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                        min_holes=1, fill_value=0, p=0.3),
        # ── Normalize & Tensor ──────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────
# CUSTOM DATASET
# ─────────────────────────────────────────────────────────────
class PneumoniaDataset(Dataset):
    """
    Loads chest X-ray images from:
        root/
          NORMAL/   *.jpeg
          PNEUMONIA/*.jpeg
    """
    CLASS_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, root_dir: str, transform=None, return_path: bool = False):
        self.root_dir    = root_dir
        self.transform   = transform
        self.return_path = return_path
        self.samples     = []
        self._load_samples()

    def _load_samples(self):
        for cls_name, label in self.CLASS_MAP.items():
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), label))
        print(f"  📂 Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self._enhance_xray(img)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if self.return_path:
            return img, label, path
        return img, label

    @staticmethod
    def _enhance_xray(img: np.ndarray) -> np.ndarray:
        """
        Medical-grade preprocessing:
        1. Bilateral denoise (preserve edges)
        2. CLAHE on luminance channel
        """
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return img

# ─────────────────────────────────────────────────────────────
# HELPER: build dataset from a sample list (no disk scan)
# ─────────────────────────────────────────────────────────────
def _dataset_from_samples(samples, transform, return_path=False):
    """Create a PneumoniaDataset from a pre-built sample list."""
    ds = PneumoniaDataset.__new__(PneumoniaDataset)
    ds.root_dir    = ""
    ds.transform   = transform
    ds.return_path = return_path
    ds.samples     = samples
    return ds

# ─────────────────────────────────────────────────────────────
# CLASS WEIGHTS & SAMPLER
# ─────────────────────────────────────────────────────────────
def get_class_weights(dataset: PneumoniaDataset) -> torch.Tensor:
    labels  = [s[1] for s in dataset.samples]
    counts  = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    print(f"  ⚖️  Class weights → NORMAL: {weights[0]:.4f} | PNEUMONIA: {weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float32)

def get_weighted_sampler(dataset: PneumoniaDataset) -> WeightedRandomSampler:
    labels         = [s[1] for s in dataset.samples]
    counts         = np.bincount(labels)
    sample_weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(
        weights    = sample_weights,
        num_samples= len(sample_weights),
        replacement= True
    )

# ─────────────────────────────────────────────────────────────
# DATALOADER FACTORY  — proper stratified split
# ─────────────────────────────────────────────────────────────
def build_dataloaders(train_dir, val_dir, test_dir, cfg=TRAIN_CONFIG):
    """
    The Kaggle chest-xray dataset ships with only 16 val images,
    which makes AUC=1.0 meaningless and triggers false early-stopping.

    Fix: merge the tiny val folder into train, then do a proper
    stratified 85/15 split so validation is representative.
    """
    print("\n📦 Building datasets (merging val → train for proper split)...")

    # ── Collect all train + val samples ──────────────────────
    all_samples = []
    for folder in [train_dir, val_dir]:
        for cls_name, label in PneumoniaDataset.CLASS_MAP.items():
            cls_dir = os.path.join(folder, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    all_samples.append((os.path.join(cls_dir, fname), label))

    # ── Stratified split ─────────────────────────────────────
    labels_all = [s[1] for s in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size    = cfg["val_split"],   # 0.15 → ~780 val images
        stratify     = labels_all,
        random_state = cfg["seed"],
    )

    n_train_normal   = sum(1 for _, l in train_samples if l == 0)
    n_train_pneumo   = sum(1 for _, l in train_samples if l == 1)
    n_val_normal     = sum(1 for _, l in val_samples   if l == 0)
    n_val_pneumo     = sum(1 for _, l in val_samples   if l == 1)

    print(f"  📂 Train → NORMAL: {n_train_normal} | PNEUMONIA: {n_train_pneumo} "
          f"| Total: {len(train_samples)}")
    print(f"  📂 Val   → NORMAL: {n_val_normal}   | PNEUMONIA: {n_val_pneumo} "
          f"| Total: {len(val_samples)}")

    # ── Build datasets ───────────────────────────────────────
    train_ds = _dataset_from_samples(train_samples, get_train_transforms())
    val_ds   = _dataset_from_samples(val_samples,   get_val_transforms())
    test_ds  = PneumoniaDataset(test_dir, transform=get_val_transforms(),
                                return_path=True)

    sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size         = cfg["batch_size"],
        sampler            = sampler,
        num_workers        = cfg["num_workers"],   # 4
        pin_memory         = cfg["pin_memory"],
        drop_last          = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = cfg["batch_size"] * 2,
        shuffle            = False,
        num_workers        = cfg["num_workers"],
        pin_memory         = cfg["pin_memory"],
        persistent_workers = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 0,       # 0 = main process only — fixes Windows spawn crash
        pin_memory  = False,
    )

    class_weights = get_class_weights(train_ds)
    print(f"  ✅ Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_weights