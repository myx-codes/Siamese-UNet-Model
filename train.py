# train.py
import os
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast 
import imageio.v3 as imageio
import yaml

# --- LOCAL IMPORTS ---
from model.siamese_unet import SiameseUNetSCJAG
from model.losses import SiameseLoss          
from modules.dataset import CT2DDatasetPairs
from modules.utils import initialize_split_file

# ============================== LOG & UTILS ==============================
def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def set_seed(seed: int = 2027):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ============================== EMA ==============================
class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1.0 - self.decay) * p.detach()

# ============================== METRICS ==============================
@torch.no_grad()
def dice_coeff(pred, target, smooth=1e-6):
    """Calculates Dice coefficient metric for validation only."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()

# ============================== MAIN ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 2027))
    log("=== TRAINING STARTED (SIAMESE MODE) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Output Directories
    exp_name = time.strftime("run_%Y%m%d_%H%M")
    exp_dir = os.path.join(cfg["paths"]["outputs_root"], "runs", exp_name)
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    debug_dir = os.path.join(exp_dir, "debug")
    ensure_dirs([exp_dir, ckpt_dir, debug_dir])

    # 3. Data Splits
    splits_path = cfg["paths"]["splits_json"]
    if not os.path.exists(splits_path):
        initialize_split_file(cfg)
    
    with open(splits_path) as f:
        split = json.load(f)

    # 4. Datasets & Loaders
    bs = cfg["train"]["batch_size"]
    
    train_dataset = CT2DDatasetPairs(
        patient_ids=split["train"],
        dicom_root=cfg["paths"]["dicom_root"],
        mask_root=cfg["paths"]["mask_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        hu_windows=cfg["data"]["hu_windows"],
        z_context=cfg["data"]["z_context"],
        pair_mode=cfg["data"]["pair_mode"],
        augment=True
    )
    
    val_dataset = CT2DDatasetPairs(
        patient_ids=split["val"],
        augment=False,
        dicom_root=cfg["paths"]["dicom_root"],
        mask_root=cfg["paths"]["mask_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        hu_windows=cfg["data"]["hu_windows"],
        z_context=cfg["data"]["z_context"],
        pair_mode=cfg["data"]["pair_mode"]
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    log(f"Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 5. Load Model
    model = SiameseUNetSCJAG(config=cfg).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 6. Loss & Optimizer
    criterion = SiameseLoss(cfg).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    ema = EMA(model, decay=cfg["train"]["ema_decay"])

    best_dice = 0.0
    epochs = cfg["train"]["epochs"]

    # ============================== LOOP ==============================
    for epoch in range(1, epochs + 1):
        model.train()
        losses = {'total': [], 'seg': [], 'adv': []}
        train_dices = []

        # GRL Alpha calculation (if domain adversarial is used)
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        
        for xin, yin, xex, yex, _ in train_loader:
            xin, yin = xin.to(device, non_blocking=True), yin.to(device, non_blocking=True)
            xex, yex = xex.to(device, non_blocking=True), yex.to(device, non_blocking=True)

            with autocast():
                # Forward Pass
                out = model(xin, xex, alpha=alpha) 
                
                # Loss Calculation (Logic from Losses.py)
                targets = {'in': yin, 'ex': yex}
                loss, loss_dict = criterion(out, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            # Metrics Logging
            losses['total'].append(loss.item())
            losses['seg'].append(loss_dict['seg'].item())
            losses['adv'].append(loss_dict['adv'].item())

            with torch.no_grad():
                train_dices.append(dice_coeff(out['pred_mask_in'], yin))

        scheduler.step()

        # ============================== VALIDATION ==============================
        model.eval()
        val_dices = []
        with torch.no_grad():
            for i, (xin, yin, xex, yex, _) in enumerate(val_loader):
                xin, yin = xin.to(device), yin.to(device)
                xex, yex = xex.to(device), yex.to(device)

                out = model(xin, xex)
                pred_in = out['pred_mask_in'] 

                # Resize 
                if pred_in.shape[2:] != yin.shape[2:]:
                     pred_in = torch.nn.functional.interpolate(pred_in, size=yin.shape[2:], mode='bilinear')

                val_dices.append(dice_coeff(pred_in, yin))

                # Visual Debug
                if i == 0 and epoch % 5 == 0:
                    img = xin[0].mean(0).cpu().numpy()
                    gt  = yin[0,0].cpu().numpy()
                    pr  = torch.sigmoid(pred_in[0,0]).cpu().numpy()
                    
                    imageio.imwrite(f"{debug_dir}/e{epoch:03d}_img.png", (np.clip(img, 0, 1)*255).astype(np.uint8))
                    imageio.imwrite(f"{debug_dir}/e{epoch:03d}_gt.png",  (gt*255).astype(np.uint8))
                    imageio.imwrite(f"{debug_dir}/e{epoch:03d}_pr.png",  (pr*255).astype(np.uint8))

        avg_loss = sum(losses['total']) / len(losses['total'])
        avg_train_dice = sum(train_dices) / len(train_dices)
        avg_val_dice = sum(val_dices) / len(val_dices) if val_dices else 0.0

        log(f"[Ep {epoch:03d}] Loss: {avg_loss:.4f} (Seg:{np.mean(losses['seg']):.3f} Adv:{np.mean(losses['adv']):.3f}) | "
            f"TrDice: {avg_train_dice:.4f} | ValDice: {avg_val_dice:.4f}")

        # Save Best
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'config': cfg,
                'dice': best_dice
            }, os.path.join(ckpt_dir, "best_model.pth"))
            log(f"*** Best Model Saved (Dice: {best_dice:.4f}) ***")

if __name__ == "__main__":
    main()