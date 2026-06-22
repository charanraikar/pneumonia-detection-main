# ddpm.py - Denoising Diffusion Probabilistic Model for X-ray synthesis
"""
DDPM Reference: Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
Fix: Decoder ResBlocks now receive in_ch = up_ch + skip_ch (concatenated channels).
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from config import DDPM_CONFIG, DDPM_DIR

# ─────────────────────────────────────────────────────────────
# SINUSOIDAL TIME EMBEDDING
# ─────────────────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

# ─────────────────────────────────────────────────────────────
# U-NET BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        # num_groups must divide in_ch — use min(8, in_ch) as safe default
        g1 = min(8, in_ch)
        g2 = min(8, out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
        self.block1   = nn.Sequential(
            nn.GroupNorm(g1, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2   = nn.Sequential(
            nn.GroupNorm(g2, out_ch), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h            = self.block1(x)
        scale, shift = self.time_mlp(t).chunk(2, dim=1)
        h            = h * (scale[..., None, None] + 1) + shift[..., None, None]
        h            = self.block2(h)
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g         = min(8, ch)
        self.norm = nn.GroupNorm(g, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h          = self.norm(x)
        q, k, v    = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(attn)

# ─────────────────────────────────────────────────────────────
# U-NET  (fixed decoder channel arithmetic)
# ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    Compact U-Net for 64x64 grayscale X-ray denoising.

    Channel sizes (base_ch=64):
        chs = [64, 128, 256, 512]

    Encoder skip shapes (input 64x64):
        e1 : (B, 128, 32, 32)
        e2 : (B, 256, 16, 16)
        e3 : (B, 512,  8,  8)

    Decoder after upsample + cat(skip):
        dec3 in_ch = 256 (up3 out) + 512 (e3) = 768 -> out 256
        dec2 in_ch = 128 (up2 out) + 256 (e2) = 384 -> out 128
        dec1 in_ch =  64 (up1 out) + 128 (e1) = 192 -> out  64
    """
    def __init__(self, img_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8]   # [64,128,256,512]

        # ── Time embedding ───────────────────────────────────
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── Encoder ──────────────────────────────────────────
        self.enc_in = nn.Conv2d(img_ch, chs[0], 3, padding=1)
        self.enc1   = ResBlock(chs[0], chs[1], time_dim)          # -> (B,128,H,W)
        self.down1  = nn.Conv2d(chs[1], chs[1], 4, 2, 1)          # -> (B,128,H/2)
        self.enc2   = ResBlock(chs[1], chs[2], time_dim)          # -> (B,256,H/2)
        self.down2  = nn.Conv2d(chs[2], chs[2], 4, 2, 1)          # -> (B,256,H/4)
        self.enc3   = ResBlock(chs[2], chs[3], time_dim)          # -> (B,512,H/4)
        self.down3  = nn.Conv2d(chs[3], chs[3], 4, 2, 1)          # -> (B,512,H/8)

        # ── Bottleneck ───────────────────────────────────────
        self.mid1     = ResBlock(chs[3], chs[3], time_dim)
        self.mid_attn = AttentionBlock(chs[3])
        self.mid2     = ResBlock(chs[3], chs[3], time_dim)

        # ── Decoder (in_ch = upsample_out + skip) ────────────
        self.up3  = nn.ConvTranspose2d(chs[3], chs[2], 4, 2, 1)
        self.dec3 = ResBlock(chs[2] + chs[3], chs[2], time_dim)   # 256+512=768->256

        self.up2  = nn.ConvTranspose2d(chs[2], chs[1], 4, 2, 1)
        self.dec2 = ResBlock(chs[1] + chs[2], chs[1], time_dim)   # 128+256=384->128

        self.up1  = nn.ConvTranspose2d(chs[1], chs[0], 4, 2, 1)
        self.dec1 = ResBlock(chs[0] + chs[1], chs[0], time_dim)   #  64+128=192-> 64

        # ── Output ───────────────────────────────────────────
        g_out    = min(8, chs[0])
        self.out = nn.Sequential(
            nn.GroupNorm(g_out, chs[0]), nn.SiLU(),
            nn.Conv2d(chs[0], img_ch, 1)
        )

    def forward(self, x, t):
        te = self.time_emb(t)

        # Encoder
        x0 = self.enc_in(x)
        e1 = self.enc1(x0, te)                    # (B,128,32,32)
        e2 = self.enc2(self.down1(e1), te)         # (B,256,16,16)
        e3 = self.enc3(self.down2(e2), te)         # (B,512, 8, 8)
        m  = self.down3(e3)                        # (B,512, 4, 4)

        # Bottleneck
        m  = self.mid2(self.mid_attn(self.mid1(m, te)), te)

        # Decoder — upsample then cat skip, then ResBlock
        d3 = self.dec3(torch.cat([self.up3(m),  e3], dim=1), te)  # 768->256
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), te)  # 384->128
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), te)  # 192-> 64

        return self.out(d1)

# ─────────────────────────────────────────────────────────────
# DDPM DIFFUSION ENGINE
# ─────────────────────────────────────────────────────────────
class DDPM:
    def __init__(self, cfg=DDPM_CONFIG, device="cuda"):
        self.cfg    = cfg
        self.device = device
        self.T      = cfg["timesteps"]
        self._build_schedule()

    def _build_schedule(self):
        if self.cfg["schedule"] == "cosine":
            s     = 0.008
            steps = self.T + 1
            t     = torch.linspace(0, self.T, steps) / self.T
            f     = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = torch.clamp(1 - alphas_cumprod[1:] / alphas_cumprod[:-1],
                                0.0, 0.9999)
        else:
            betas = torch.linspace(self.cfg["beta_start"],
                                   self.cfg["beta_end"], self.T)

        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, 0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_var  = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        def reg(t): return t.float().to(self.device)
        self.betas            = reg(betas)
        self.sqrt_alpha_bar   = reg(alpha_bar.sqrt())
        self.sqrt_one_m_ab    = reg((1 - alpha_bar).sqrt())
        self.posterior_var    = reg(posterior_var)
        self.recip_sqrt_alpha = reg((1.0 / alphas).sqrt())
        self.betas_over_sqrt  = reg(betas / (1 - alpha_bar).sqrt())

    # ── Forward process (add noise) ──────────────────────────
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        B      = x0.shape[0]
        sq_ab  = self.sqrt_alpha_bar[t].view(B, 1, 1, 1)
        sq_1ab = self.sqrt_one_m_ab[t].view(B, 1, 1, 1)
        return sq_ab * x0 + sq_1ab * noise, noise

    # ── Single reverse step ──────────────────────────────────
    @torch.no_grad()
    def p_sample(self, model, x, t_int):
        t_tensor = torch.full((x.shape[0],), t_int,
                              device=self.device, dtype=torch.long)
        eps_pred = model(x, t_tensor)
        coef     = self.betas_over_sqrt[t_int]
        recip_sq = self.recip_sqrt_alpha[t_int]
        mean     = recip_sq * (x - coef * eps_pred)
        if t_int > 0:
            noise = torch.randn_like(x)
            var   = self.posterior_var[t_int].sqrt()
            return mean + var * noise
        return mean

    # ── Full reverse chain (generation) ─────────────────────
    @torch.no_grad()
    def sample(self, model, n_samples, img_size, channels):
        model.eval()
        x = torch.randn(n_samples, channels, img_size, img_size,
                        device=self.device)
        for t in tqdm(reversed(range(self.T)),
                      desc="Sampling", total=self.T, leave=False):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)

    # ── Training step ────────────────────────────────────────
    def train_step(self, model, x0):
        t     = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        pred  = model(xt, t)
        return F.mse_loss(pred, noise)

# ─────────────────────────────────────────────────────────────
# DDPM TRAINER
# ─────────────────────────────────────────────────────────────
def train_ddpm(train_loader, device, cfg=DDPM_CONFIG):
    print("\n🌊 Training DDPM for X-ray synthesis...")
    unet   = UNet(img_ch=cfg["channels"]).to(device)
    ddpm   = DDPM(cfg, device)
    opt    = torch.optim.AdamW(unet.parameters(), lr=cfg["lr"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    scaler = GradScaler()
    resize = torch.nn.functional.interpolate

    best_loss = float("inf")

    for epoch in range(cfg["epochs"]):
        unet.train()
        total_loss = 0

        for imgs, _ in tqdm(train_loader,
                             desc=f"DDPM Ep {epoch+1}/{cfg['epochs']}",
                             leave=False):
            imgs = resize(imgs.to(device),
                          size=(cfg["image_size"], cfg["image_size"]),
                          mode="bilinear", align_corners=False)
            imgs = imgs.mean(dim=1, keepdim=True)   # RGB -> grayscale (B,1,64,64)
            imgs = imgs * 2 - 1                      # [0,1] -> [-1,1]

            opt.zero_grad()
            with autocast():
                loss = ddpm.train_step(unet, imgs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        sched.step()
        print(f"  DDPM Epoch {epoch+1:03d} | Loss: {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(unet.state_dict(),
                       os.path.join(DDPM_DIR, "ddpm_best.pth"))

    # Generate and save samples
    unet.load_state_dict(torch.load(os.path.join(DDPM_DIR, "ddpm_best.pth")))
    samples = ddpm.sample(unet, cfg["num_generate"],
                          cfg["image_size"], cfg["channels"])
    _save_ddpm_grid(samples, os.path.join(DDPM_DIR, "generated_xrays.png"))
    print(f"✅ DDPM done. Outputs saved to {DDPM_DIR}/")
    return unet, ddpm


def _save_ddpm_grid(samples, path, nrow=4):
    imgs  = (samples.cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    n     = len(imgs)
    ncols = nrow
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("DDPM Generated Chest X-rays", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Generated grid saved: {path}")

# ─────────────────────────────────────────────────────────────
# DIFFUSION PROCESS VISUALISATION
# ─────────────────────────────────────────────────────────────
def visualise_diffusion_process(image_tensor, ddpm: DDPM, save_path: str,
                                 timesteps_to_show=(0, 100, 250, 500, 750, 999)):
    img        = image_tensor.unsqueeze(0).to(ddpm.device)
    img_gray   = img.mean(dim=1, keepdim=True)
    img_scaled = F.interpolate(
        img_gray,
        (DDPM_CONFIG["image_size"], DDPM_CONFIG["image_size"]),
        mode="bilinear", align_corners=False
    ) * 2 - 1

    n    = len(timesteps_to_show)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 2.5, 3))

    orig_np = ((img_scaled[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
    axes[0].imshow(orig_np, cmap="gray")
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for i, t_val in enumerate(timesteps_to_show):
        t_tensor  = torch.tensor([t_val], device=ddpm.device)
        noisy, _  = ddpm.q_sample(img_scaled, t_tensor)
        noisy_np  = ((noisy[0, 0].cpu().numpy() + 1) / 2).clip(0, 1)
        axes[i + 1].imshow(noisy_np, cmap="gray")
        axes[i + 1].set_title(f"t={t_val}", fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("DDPM Forward Noising Process", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Diffusion process saved: {save_path}")