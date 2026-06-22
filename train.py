# train.py - Full training pipeline with AMP, mixup, early stopping, metrics
import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
import seaborn as sns

from config import TRAIN_CONFIG, CHECKPOINT_DIR, LOGS_DIR, RESULTS_DIR
from model import build_model
from dataset import build_dataloaders

# ─────────────────────────────────────────────────────────────
# MIXUP AUGMENTATION  (powerful anti-overfitting tool)
# ─────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    """Apply mixup: blend two random samples and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=x.device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    y_a, y_b   = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ─────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULER  (cosine with warm-up)
# ─────────────────────────────────────────────────────────────
def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch
    total_steps  = cfg["epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─────────────────────────────────────────────────────────────
# TRAIN EPOCH
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler,
                scaler, device, cfg, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=95)
    for step, (imgs, labels) in enumerate(pbar):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Mixup ────────────────────────────────────────────
        if use_mixup and np.random.rand() < 0.5:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
            with autocast(enabled=cfg["mixed_precision"]):
                logits = model(imgs)
                loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
                loss   = loss / cfg["accumulation_steps"]
        else:
            with autocast(enabled=cfg["mixed_precision"]):
                logits = model(imgs)
                loss   = criterion(logits, labels) / cfg["accumulation_steps"]

        scaler.scale(loss).backward()

        if (step + 1) % cfg["accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * cfg["accumulation_steps"]
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}",
                         acc=f"{correct/total:.3f}")

    return total_loss / len(loader), correct / total

# ─────────────────────────────────────────────────────────────
# VALIDATION EPOCH
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, cfg):
    model.eval()
    total_loss                        = 0.0
    all_preds, all_labels, all_probs  = [], [], []

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=95):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=cfg["mixed_precision"]):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss  += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc, all_preds, all_labels, all_probs

# ─────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train(train_dir, val_dir, test_dir, device, cfg=TRAIN_CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = \
        build_dataloaders(train_dir, val_dir, test_dir, cfg)

    # ── Model ────────────────────────────────────────────────
    model, criterion, optimizer = build_model(device, class_weights)

    # ── Phase 1: Warm-up (backbone frozen) ───────────────────
    model.freeze_backbone()
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg["warmup_epochs"]
    )

    scaler  = GradScaler(enabled=cfg["mixed_precision"])
    writer  = SummaryWriter(LOGS_DIR)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [],
               "train_acc": []}

    best_auc      = 0.0
    patience_left = cfg["early_stopping"]

    print(f"\n{'='*65}")
    print(f"  🚀 Training ConvNeXt-Base — up to {cfg['epochs']} epochs")
    print(f"  Device    : {device}")
    print(f"  Batch     : {cfg['batch_size']} | AMP: {cfg['mixed_precision']}")
    print(f"  Val split : {int(cfg['val_split']*100)}% stratified")
    print(f"  Mixup     : α=0.4 (50% of batches)")
    print(f"  Patience  : {cfg['early_stopping']} epochs")
    print(f"{'='*65}\n")

    fine_tune_started = False

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ── Switch to full fine-tune after warm-up ────────────
        if epoch == cfg["warmup_epochs"] + 1 and not fine_tune_started:
            model.unfreeze_backbone()
            scheduler = build_scheduler(optimizer, cfg, len(train_loader))
            fine_tune_started = True
            print("  🔓 Full fine-tuning started.\n")
        elif not fine_tune_started:
            scheduler = warmup_scheduler

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device, cfg,
            use_mixup=(fine_tune_started)   # mixup only during fine-tuning
        )
        val_loss, val_acc, val_auc, v_preds, v_labels, v_probs = \
            eval_epoch(model, val_loader, criterion, device, cfg)

        elapsed = time.time() - t0
        gap     = train_acc - val_acc

        print(f"  Ep {epoch:03d}/{cfg['epochs']} | "
              f"Train {train_loss:.4f}/{train_acc:.3f} | "
              f"Val {val_loss:.4f}/{val_acc:.3f} AUC:{val_auc:.4f} | "
              f"Gap:{gap:+.3f} | {elapsed:.0f}s")

        # TensorBoard
        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("AUC/val",   val_auc, epoch)
        writer.add_scalar("LR",        optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train-Val-Gap", gap, epoch)

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["train_acc"].append(train_acc)

        # Checkpoint on best AUC
        if val_auc > best_auc:
            best_auc      = val_auc
            patience_left = cfg["early_stopping"]
            ckpt_path     = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_auc"     : val_auc,
                "val_acc"     : val_acc,
            }, ckpt_path)
            print(f"  ✅ New best AUC: {best_auc:.4f} — checkpoint saved")
        else:
            patience_left -= 1
            print(f"  ⏳ No improvement. Patience: {patience_left}/{cfg['early_stopping']}")
            if patience_left == 0:
                print(f"\n⏹️  Early stopping at epoch {epoch}. Best AUC: {best_auc:.4f}")
                break

        torch.cuda.empty_cache()

    writer.close()
    print(f"\n{'='*65}")
    print(f"  Training complete. Best Val AUC: {best_auc:.4f}")
    print(f"{'='*65}\n")

    _plot_history(history)

    # ── Final test evaluation ─────────────────────────────────
    print("📊 Running final test evaluation...")
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state"])
    evaluate_test(model, test_loader, device, cfg)

    return model, test_loader

# ─────────────────────────────────────────────────────────────
# TEST EVALUATION
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model, test_loader, device, cfg=TRAIN_CONFIG):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    CLASS_NAMES = cfg["classes"]

    for imgs, labels, _ in tqdm(test_loader, desc="  Test ", leave=False):
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n📋 Test Results:")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")
    print("\n" + classification_report(all_labels, all_preds,
                                       target_names=CLASS_NAMES))

    _plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES)
    _plot_roc_curve(all_labels, all_probs, auc)

    report = {
        "accuracy": round(acc, 4),
        "auc"     : round(auc, 4),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
        ),
    }
    out = os.path.join(RESULTS_DIR, "test_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📄 Results saved: {out}")

# ─────────────────────────────────────────────────────────────
# PLOT UTILITIES
# ─────────────────────────────────────────────────────────────
def _plot_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Training History", fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train", color="#3498db")
    axes[0].plot(history["val_loss"],   label="Val",   color="#e74c3c")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="#3498db")
    axes[1].plot(history["val_acc"],   label="Val",   color="#e74c3c")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_auc"], color="#9b59b6")
    axes[2].set_title("Val AUC-ROC"); axes[2].grid(alpha=0.3)

    # Overfitting gap
    gap = [ta - va for ta, va in
           zip(history["train_acc"], history["val_acc"])]
    axes[3].plot(gap, color="#e67e22")
    axes[3].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[3].set_title("Train-Val Accuracy Gap"); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Training curves saved: {path}")

def _plot_confusion_matrix(labels, preds, class_names):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved: {path}")

def _plot_roc_curve(labels, probs, auc):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 ROC curve saved: {path}")