import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.lines import Line2D

# modules
from model.siamese_unet import SiameseUNetSCJAG
from modules.dataset import CT2DDatasetPairs


# =========================================================
# 1. Helper Functions
# =========================================================

def _calc_iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate IoU on numpy binary arrays."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + 1e-6) / (union + 1e-6)


def calculate_metrics(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute Dice and IoU for a single pair (inhale or exhale)."""
    probs = torch.sigmoid(pred_logits)
    pred_mask = (probs > threshold).float()
    
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return dice.item(), iou.item()


def find_latest_checkpoint(config: dict) -> str | None:
    """Automatically find the most recent best checkpoint."""
    out_root = config.get("paths", {}).get("outputs_root", "outputs")
    runs_dir = os.path.join(out_root, "runs")
    if not os.path.exists(runs_dir):
        return None
    
    all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))]
    if not all_runs:
        return None
    
    all_runs.sort(key=os.path.getmtime, reverse=True)
    latest_run = all_runs[0]
    
    candidates = ["best_model.pth", "best.pth"]
    for name in candidates:
        ckpt_path = os.path.join(latest_run, "ckpt", name)
        if os.path.exists(ckpt_path):
            print(f"AUTO-DETECTED CHECKPOINT: {ckpt_path}")
            return ckpt_path
    return None


def get_save_directory(ckpt_path: str) -> str:
    """Create output directory based on checkpoint run name."""
    base_output = "/workexternal/muhammad/Unet_Model251104/outputs/test_results"
    parts = ckpt_path.split(os.sep)
    run_name = None
    for p in parts:
        if p.startswith("run_"):
            run_name = p
            break
    
    if run_name:
        final_dir = os.path.join(base_output, run_name)
    else:
        now = datetime.now()
        final_dir = os.path.join(base_output, f"manual_test_{now.strftime('%Y%m%d_%H%M')}")
    
    os.makedirs(final_dir, exist_ok=True)
    return final_dir


def compute_patient_summary(df_all: pd.DataFrame, trim_head_pct: float = 0, trim_tail_pct: float = 0) -> pd.DataFrame:
    """Compute per-patient mean metrics, optionally trimming head/tail slices."""
    final_stats = []
    for pid in df_all['patient_id'].unique():
        sub = df_all[df_all['patient_id'] == pid].sort_values("slice_idx")
        n = len(sub)
        start = int(n * trim_head_pct / 100)
        end = int(n * (100 - trim_tail_pct) / 100)
        if start >= end:
            start, end = 0, n
        sub_trimmed = sub.iloc[start:end]
        
        mean_vals = sub_trimmed.mean(numeric_only=True)
        mean_vals["patient_id"] = pid
        mean_vals["n_total"] = n
        mean_vals["n_used"] = len(sub_trimmed)
        final_stats.append(mean_vals)
    
    return pd.DataFrame(final_stats)


# =========================================================
# 2. Visualization Functions
# =========================================================

def save_overlay_image(img_tensor: torch.Tensor, gt_tensor: torch.Tensor, pred_tensor: torch.Tensor,
                       save_path: str, title: str = "", iou_value: float | None = None):
    """Save three-panel overlay: Ground Truth (Red), Prediction (Blue), Combined."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    if gt_tensor.dim() == 4:
        gt_tensor = gt_tensor.squeeze(0)
    if pred_tensor.dim() == 4:
        pred_tensor = pred_tensor.squeeze(0)

    mid_idx = img_tensor.shape[0] // 2
    dicom_gray01 = img_tensor[mid_idx].cpu().numpy()
    gt_slice = gt_tensor.squeeze().cpu().numpy()
    pred_slice = torch.sigmoid(pred_tensor).squeeze().cpu().numpy()

    img = np.asarray(dicom_gray01, dtype=float)
    gt = (np.asarray(gt_slice) > 0)
    pred_binary = (np.asarray(pred_slice) > 0.5)

    if iou_value is None:
        iou_value = _calc_iou_np(pred_binary, gt)

    # Robust normalization
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    img01 = (img - p1) / (p99 - p1 + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#bfbfbf')
    
    full_title = f"{title} | IoU={iou_value:.3f}" if title else f"IoU={iou_value:.3f}"
    fig.suptitle(full_title, fontsize=16, color='black', fontweight='bold')

    # Ground Truth
    axes[0].imshow(img01, cmap='gray', vmin=0, vmax=1)
    masked_gt = np.ma.masked_where(gt == 0, gt)
    axes[0].imshow(masked_gt, cmap='Reds', alpha=0.35, vmin=0, vmax=1)
    axes[0].set_title("Ground Truth (Red)", color='black')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(img01, cmap='gray', vmin=0, vmax=1)
    masked_pr = np.ma.masked_where(pred_binary == 0, pred_binary)
    axes[1].imshow(masked_pr, cmap='Blues', alpha=0.35, vmin=0, vmax=1)
    axes[1].set_title("Prediction (Blue)", color='black')
    axes[1].axis('off')

    # Combined
    axes[2].imshow(img01, cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(masked_gt, cmap='Reds', alpha=0.35, vmin=0, vmax=1)
    axes[2].imshow(masked_pr, cmap='Blues', alpha=0.35, vmin=0, vmax=1)
    axes[2].set_title("Combined Overlay", color='black')
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_single_patient_plot(df_patient: pd.DataFrame, save_path: str, patient_id: str):
    """Plot Dice and IoU curves for inhale/exhale on a single patient."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    df_sorted = df_patient.sort_values(by="slice_idx")
    z = df_sorted["slice_idx"]
    
    plt.plot(z, df_sorted["dice_in"], color="#1f77b4", linestyle='--', linewidth=2, label="Dice (Inhale)")
    plt.plot(z, df_sorted["dice_ex"], color="#1f77b4", linestyle='-', linewidth=2, label="Dice (Exhale)")
    plt.plot(z, df_sorted["iou_in"], color="#d62728", linestyle='--', linewidth=2, label="IoU (Inhale)")
    plt.plot(z, df_sorted["iou_ex"], color="#d62728", linestyle='-', linewidth=2, label="IoU (Exhale)")

    avg_dice = (df_sorted["dice_in"].mean() + df_sorted["dice_ex"].mean()) / 2
    avg_iou = (df_sorted["iou_in"].mean() + df_sorted["iou_ex"].mean()) / 2

    plt.title(f"Patient {patient_id} | Avg Dice: {avg_dice:.4f} | Avg IoU: {avg_iou:.4f}",
              fontsize=15, fontweight='bold')
    plt.xlabel("Slice Index", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_global_plot(df_all: pd.DataFrame, save_path: str, metric: str = 'dice', title_prefix: str = "FULL MODEL"):
    """Global plot across all patients (anonymized). Inhale = dashed, Exhale = solid."""
    pids = df_all['patient_id'].unique()
    if len(pids) == 0:
        return
    
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, len(pids))]
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Dummy lines for phase legend
    style_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='--'),
        Line2D([0], [0], color='black', lw=2, linestyle='-')
    ]
    
    for i, pid in enumerate(pids):
        anon_label = f"P{i+1}"
        color = colors[i]
        
        df_p = df_all[df_all['patient_id'] == pid].sort_values('slice_idx')
        
        # Inhale (dashed)
        val_in = df_p[f'{metric}_in']
        plt.plot(df_p['slice_idx'], val_in, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        
        # Exhale (solid)
        val_ex = df_p[f'{metric}_ex']
        plt.plot(df_p['slice_idx'], val_ex, color=color, linestyle='-', linewidth=1.5, alpha=0.8)
    
    metric_name = "Dice Score" if metric == 'dice' else "IoU Score"
    plt.xlabel("Slice Index", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.ylim(0.0, 1.02)
    
    # Patient legend
    leg1 = plt.legend(ncol=10, loc='lower center', bbox_to_anchor=(0.5, -0.25),
                      fontsize=9, title="Patients (Anonymized)", frameon=True)
    
    # Phase legend
    plt.legend(style_lines, ['Inhale (---)', 'Exhale (-)'], loc='lower right',
               fontsize=10, framealpha=0.9, title="Phases")
    
    plt.gca().add_artist(leg1)
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Global {metric.upper()} plot saved: {save_path}")


# =========================================================
# 3. Patient-Level Processing and Visualization
# =========================================================

def process_patient_visuals(patient_data: list, output_dir: str, config: dict):
    """Process metrics, plots, and overlay images for one patient."""
    pid = patient_data[0]['pid']
    
    # Directories
    patient_dir = os.path.join(output_dir, pid)
    overlay_dir = os.path.join(patient_dir, "overlay")
    os.makedirs(patient_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Clean DataFrame (no tensors)
    csv_data = []
    for item in patient_data:
        csv_data.append({
            'patient_id': item['pid'],
            'slice_idx': item['slice_idx'],
            'dice_in': item['dice_in'],
            'iou_in': item['iou_in'],
            'dice_ex': item['dice_ex'],
            'iou_ex': item['iou_ex']
        })
    df = pd.DataFrame(csv_data)
    
    # Save metrics
    df.to_csv(os.path.join(patient_dir, "metrics_full.csv"), index=False)
    
    # Single patient plot
    save_single_patient_plot(df, os.path.join(patient_dir, f"{pid}_metrics.png"), pid)
    
    # Trimmed metrics (if configured)
    eval_cfg = config.get("eval", {})
    trim_head = float(eval_cfg.get("trim_head_pct", 10))
    trim_tail = float(eval_cfg.get("trim_tail_pct", 20))
    
    df_sorted = df.sort_values("slice_idx")
    n = len(df_sorted)
    start = int(n * trim_head / 100)
    end = int(n * (100 - trim_tail) / 100)
    if start >= end:
        start, end = 0, n
    
    df_trim = df_sorted.iloc[start:end]
    df_trim.to_csv(os.path.join(patient_dir, "metrics_trim.csv"), index=False)

    # Overlay images
    do_overlays = eval_cfg.get("overlays", "sampled")
    if do_overlays == "none":
        return

    sample_k = int(eval_cfg.get("overlays_sample_k", 4))
    indices = list(range(len(patient_data)))
    
    if do_overlays == "sampled" and len(indices) > sample_k:
        selected_indices = random.sample(indices, sample_k)
        selected_indices.sort()
    else:
        selected_indices = indices

    for idx in selected_indices:
        item = patient_data[idx]
        z = item['slice_idx']
        
        save_overlay_image(item['xin'], item['yin'], item['pred_in'],
                           os.path.join(overlay_dir, f"Z{z:03d}_IN.png"),
                           title=f"{pid} Inhale Z{z}", iou_value=item['dice_in'])
        
        save_overlay_image(item['xex'], item['yex'], item['pred_ex'],
                           os.path.join(overlay_dir, f"Z{z:03d}_EX.png"),
                           title=f"{pid} Exhale Z{z}", iou_value=item['dice_ex'])


# =========================================================
# 4. Main Inference & Evaluation Loop
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for Siamese Attention U-Net")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint(cfg)
    
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("Error: No valid checkpoint found.")
        return

    save_dir = get_save_directory(ckpt_path)
    print(f"Results will be saved to: {save_dir}")

    print(f"Loading model from: {ckpt_path}")
    model = SiameseUNetSCJAG(config=cfg).to(device)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        # Remove 'module.' prefix if DataParallel was used
        cleaned_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_sd)
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Load split
    splits_path = cfg["paths"]["splits_json"]
    with open(splits_path) as f:
        split_data = json.load(f)
    
    if args.split not in split_data or len(split_data[args.split]) == 0:
        print(f"Warning: '{args.split}' split is empty. Falling back to 'val'.")
        args.split = "val"
    
    patient_ids = split_data[args.split]
    print(f"\n{'='*50}")
    print(f"EVALUATION ON: {args.split.upper()} | Patients: {len(patient_ids)}")
    print(f"{'='*50}\n")

    # Dataset & DataLoader
    dataset = CT2DDatasetPairs(
        patient_ids=patient_ids,
        dicom_root=cfg["paths"]["dicom_root"],
        mask_root=cfg["paths"]["mask_root"],
        img_size=tuple(cfg["data"]["img_size"]),
        hu_windows=cfg["data"]["hu_windows"],
        z_context=cfg["data"]["z_context"],
        pair_mode=cfg["data"]["pair_mode"],
        augment=False,
        verbose=True
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Inference loop
    all_metrics = []
    current_pid = None
    patient_buffer = []

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            xin, yin, xex, yex, meta = batch
            pid = meta[0][0]           # string
            z_idx = meta[1].item()     # int
            
            # Process completed patient
            if current_pid is not None and pid != current_pid:
                process_patient_visuals(patient_buffer, save_dir, cfg)
                patient_buffer = []
            
            current_pid = pid

            xin, yin = xin.to(device), yin.to(device)
            xex, yex = xex.to(device), yex.to(device)

            out = model(xin, xex)
            pred_in = out['pred_mask_in']
            pred_ex = out['pred_mask_ex']

            # Resize predictions if needed
            if pred_in.shape[2:] != yin.shape[2:]:
                pred_in = F.interpolate(pred_in, size=yin.shape[2:], mode='bilinear', align_corners=False)
                pred_ex = F.interpolate(pred_ex, size=yex.shape[2:], mode='bilinear', align_corners=False)

            dice_in, iou_in = calculate_metrics(pred_in, yin)
            dice_ex, iou_ex = calculate_metrics(pred_ex, yex)

            record = {
                "pid": pid,
                "slice_idx": z_idx,
                "dice_in": dice_in, "iou_in": iou_in,
                "dice_ex": dice_ex, "iou_ex": iou_ex,
                "xin": xin.cpu(), "yin": yin.cpu(), "pred_in": pred_in.cpu(),
                "xex": xex.cpu(), "yex": yex.cpu(), "pred_ex": pred_ex.cpu()
            }
            patient_buffer.append(record)
            
            all_metrics.append({
                "patient_id": pid, "slice_idx": z_idx,
                "dice_in": dice_in, "iou_in": iou_in,
                "dice_ex": dice_ex, "iou_ex": iou_ex,
            })

        # Process last patient
        if patient_buffer:
            process_patient_visuals(patient_buffer, save_dir, cfg)

    # =========================================================
    # 5. Global Summary & Reporting
    # =========================================================
    df_raw = pd.DataFrame(all_metrics)
    df_raw.to_csv(os.path.join(save_dir, "metrics_slice_raw.csv"), index=False)
    
    eval_cfg = cfg.get("eval", {})
    mode = eval_cfg.get("metrics_mode", "both")
    trim_head = float(eval_cfg.get("trim_head_pct", 10))
    trim_tail = float(eval_cfg.get("trim_tail_pct", 20))

    tasks = []
    if mode in ['full', 'both']:
        tasks.append(("FULL", 0, 0))
    if mode in ['trim', 'both']:
        tasks.append(("TRIMMED", trim_head, trim_tail))

    for name, h_pct, t_pct in tasks:
        # Prepare data for plotting
        if name == "TRIMMED":
            trimmed_rows = []
            for pid in df_raw['patient_id'].unique():
                sub = df_raw[df_raw['patient_id'] == pid].sort_values("slice_idx")
                n = len(sub)
                s = int(n * h_pct / 100)
                e = int(n * (100 - t_pct) / 100)
                if s >= e:
                    s, e = 0, n
                trimmed_rows.append(sub.iloc[s:e])
            df_for_plot = pd.concat(trimmed_rows) if trimmed_rows else df_raw
        else:
            df_for_plot = df_raw

        # Global plots
        save_global_plot(df_for_plot, os.path.join(save_dir, f"global_dice_{name.lower()}.png"),
                         metric='dice', title_prefix=f"{name} MODEL")
        save_global_plot(df_for_plot, os.path.join(save_dir, f"global_iou_{name.lower()}.png"),
                         metric='iou', title_prefix=f"{name} MODEL")

        # Patient summary
        df_summary = compute_patient_summary(df_raw, trim_head_pct=h_pct, trim_tail_pct=t_pct)
        df_summary.to_csv(os.path.join(save_dir, f"patient_summary_{name.lower()}.csv"), index=False)
        
        grand_mean = df_summary.mean(numeric_only=True)
        
        print("\n" + "="*60)
        print(f" FINAL REPORT: {name} (Head trim: {h_pct}%, Tail trim: {t_pct}%)")
        print("="*60)
        print(f" Inhale  | Dice: {grand_mean['dice_in']:.4f}   IoU: {grand_mean['iou_in']:.4f}")
        print(f" Exhale  | Dice: {grand_mean['dice_ex']:.4f}   IoU: {grand_mean['iou_ex']:.4f}")
        print("-" * 60)
        print(f" Overall | Avg Dice: {(grand_mean['dice_in'] + grand_mean['dice_ex'])/2:.4f}")
        print("="*60)

    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()