# model.py - ConvNeXt architecture with custom classification head
import torch
import torch.nn as nn
import timm
from config import MODEL_CONFIG

# ─────────────────────────────────────────────────────────────
# CONVNEXT CLASSIFIER
# ─────────────────────────────────────────────────────────────
class ConvNeXtPneumonia(nn.Module):
    """
    ConvNeXt-Base backbone (ImageNet-22k pretrained) + classification head.

    Architecture:
        ConvNeXt-Base backbone
            └── Global Average Pooling  [1024]
                └── LayerNorm
                    └── Dropout(0.4)
                        └── Linear(1024 → 512)
                            └── GELU
                                └── Dropout(0.3)
                                    └── Linear(512 → num_classes)
    """
    def __init__(self, cfg=MODEL_CONFIG):
        super().__init__()
        self.num_classes = cfg["num_classes"]

        # ── Backbone ─────────────────────────────────────────
        self.backbone = timm.create_model(
            cfg["architecture"],
            pretrained     = cfg["pretrained"],
            num_classes    = 0,
            drop_path_rate = cfg["drop_path_rate"],  # 0.4
            global_pool    = "avg",
        )
        feat_dim = self.backbone.num_features   # 1024 for ConvNeXt-Base

        # ── Classification Head ───────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.40),             # increased from 0.30
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.30),             # increased from 0.20
            nn.Linear(512, self.num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("🔒 Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔓 Backbone unfrozen for fine-tuning.")

    def get_grad_cam_target_layer(self):
        return self.backbone.stages[-1]


# ─────────────────────────────────────────────────────────────
# LABEL SMOOTHING LOSS
# ─────────────────────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll       = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth    = -log_probs.mean(dim=-1)
        loss      = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        if self.weight is not None:
            w    = self.weight.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


def build_model(device, class_weights=None):
    model     = ConvNeXtPneumonia()
    model     = model.to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)

    # Differential learning rates: lower for backbone, higher for head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 5e-5,
         "weight_decay": 5e-4},
        {"params": model.head.parameters(),     "lr": 2e-4,
         "weight_decay": 5e-4},
    ])

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️  ConvNeXt-Base loaded")
    print(f"   Total params   : {total_params:,}")
    print(f"   Trainable      : {trainable:,}")
    return model, criterion, optimizer