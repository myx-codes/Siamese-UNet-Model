import csv
import os
import json
import random
import time
from typing import List

import numpy as np
import torch
from glob import glob
import yaml

# =========================================================
# 1. SYSTEM AND CONFIGURATION
# =========================================================

def log(msg: str):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def set_seed(seed: int = 2027):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dirs(paths):
    """Create directories if they do not exist."""
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_config(path: str):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(data, path):
    """Save data to a JSON file."""
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def create_csv_from_patient_list(patient_list: List[str], csv_path: str):
    """Create a CSV file containing a list of patient IDs."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id"])
        for pid in patient_list:
            writer.writerow([pid])

# =========================================================
# 2. DATA SPLITTING
# =========================================================

def check_patient_validity(pid: str, dicom_root: str, mask_root: str) -> bool:
    """
    Check whether a patient has valid DICOM series (in/ex) and corresponding masks.
    Returns True if everything is present and valid.
    """
    # 1. DICOM check
    path_in = os.path.join(dicom_root, pid, "in")
    path_ex = os.path.join(dicom_root, pid, "ex")
    
    if not (os.path.isdir(path_in) and len(glob(os.path.join(path_in, "*.dcm"))) > 0):
        return False
    if not (os.path.isdir(path_ex) and len(glob(os.path.join(path_ex, "*.dcm"))) > 0):
        return False

    # 2. Mask check
    # Inhale mask
    mask_in_ok = (
        os.path.exists(os.path.join(mask_root, pid, "in.nii.gz")) or 
        os.path.exists(os.path.join(mask_root, pid, "in", "mask.nii.gz")) or
        os.path.exists(os.path.join(mask_root, pid, "in.nii"))
    )
    # Exhale mask
    mask_ex_ok = (
        os.path.exists(os.path.join(mask_root, pid, "ex.nii.gz")) or 
        os.path.exists(os.path.join(mask_root, pid, "ex", "mask.nii.gz")) or
        os.path.exists(os.path.join(mask_root, pid, "ex.nii"))
    )

    return mask_in_ok and mask_ex_ok

def initialize_split_file(cfg: dict):
    """
    Create train/val split file.
    Called from train.py.
    """
    dicom_root = cfg["paths"]["dicom_root"]
    mask_root = cfg["paths"]["mask_root"]
    output_json = cfg["paths"]["splits_json"]
    val_percent = cfg["data"].get("val_percent", 0.2)
    seed = cfg.get("seed", 2027)

    log("Creating data split file...")
    
    if not os.path.exists(dicom_root):
        log(f"ERROR: DICOM root not found: {dicom_root}")
        return

    # Get all patient folders
    all_patients = [
        d for d in os.listdir(dicom_root) 
        if os.path.isdir(os.path.join(dicom_root, d))
    ]
    all_patients.sort()

    valid_patients = []
    skipped_patients = []

    # Validate each patient
    for pid in all_patients:
        if check_patient_validity(pid, dicom_root, mask_root):
            valid_patients.append(pid)
        else:
            skipped_patients.append(pid)

    if not valid_patients:
        raise ValueError("No valid patients found! Check your data folder structure (requires in/ex subfolders and masks).")

    log(f"Total patients found: {len(all_patients)}")
    log(f"Valid (IN + EX + Masks): {len(valid_patients)}")
    log(f"Skipped: {len(skipped_patients)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(valid_patients)

    n_val = int(len(valid_patients) * val_percent)
    if n_val == 0 and len(valid_patients) > 1:
        n_val = 1  # Ensure at least one validation patient
    
    val_ids = valid_patients[:n_val]
    train_ids = valid_patients[n_val:]

    split_data = {
        "train": train_ids,
        "val": val_ids,
        "test": [] 
    }

    save_json(split_data, output_json)
    log(f"Split saved to: {output_json}")
    log(f"Train: {len(train_ids)} | Val: {len(val_ids)}")