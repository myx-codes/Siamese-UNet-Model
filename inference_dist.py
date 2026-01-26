import os, csv
import numpy as np
import torch
import pydicom
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
import sys
from modules.preprocessing import split_dicoms_to_series
from modules.counting import counting
from modules.utils import load_config, create_csv_from_patient_list
from model.siamese_unet import SiameseUNetSCJAG

# ========================= SETTINGS =========================
# If using a single GPU, set this to 1 or 2. If using CPU only, you can set it to 4-8.
# NOTE: If you set this to 4, the model will be loaded into VRAM 4 times!
MAX_WORKERS = min(8, cpu_count()) 

# ========================= DICOM → HU =========================
def read_dicom_series(dicom_dir: str):
    """Reads a directory of DICOM files and converts them to a HU volume."""
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm")]
    if not files:
        raise RuntimeError(f"No DICOM files in {dicom_dir}")

    def _key(fp):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    files = sorted(files, key=_key)
    vol = []
    for fp in files:
        ds = pydicom.dcmread(fp, force=True)
        arr = ds.pixel_array.astype(np.int16)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        vol.append(arr * slope + inter)

    return np.stack(vol, axis=0).astype(np.float32)  # Shape: [Z,H,W]


def apply_hu_windows(vol, windows):
    """Applies multiple HU windows and normalizes to [0, 1]."""
    out = []
    for lo, hi in windows:
        ch = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        out.append(ch)
    return np.stack(out, axis=1)  # Shape: [Z,C,H,W]


# ========================= MODEL SETUP =========================
def build_siamese_model(cfg, ckpt_path, device):
    """Initializes the model and loads the checkpoint."""
    model = SiameseUNetSCJAG(config=cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(device).eval()
    return model


# ========================= INFERENCE LOGIC =========================
def infer_and_save_pngs(
    model,
    dicom_dir,
    out_dir,
    phase="in",
    img_size=(512,512),
    hu_windows=None,
    z_context=2,
    device="cuda",
    thresh=0.5,
    overlay=True,
    overlay_window=(-600,1500),
    overlay_alpha=0.6
):
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    hu_windows = hu_windows or [(-400,1600),(40,400),(-30,80),(-150,2000),(-1000,-300)]
    os.makedirs(out_dir, exist_ok=True)

    vol = read_dicom_series(dicom_dir)          # [Z,H,W]
    chw = apply_hu_windows(vol, hu_windows)     # [Z,C,H,W]
    Z, H, W = vol.shape

    def hu_to_gray(img, win):
        lo, hi = win
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    with torch.no_grad():
        for z in range(Z): 
            ctx = []
            for zz in range(z - z_context, z + z_context + 1):
                zz = max(0, min(Z - 1, zz))
                ctx.append(chw[zz])

            x = np.concatenate(ctx, axis=0)[None]  # [1,C',H,W]
            x = torch.from_numpy(x).float().to(device)

            if list(x.shape[-2:]) != list(img_size):
                x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)

            out = model(x, x, alpha=0.0)
            logit = out["pred_mask_ex"] if phase == "ex" else out["pred_mask_in"]
            prob = torch.sigmoid(logit)

            prob = F.interpolate(prob, size=(H,W), mode="bilinear", align_corners=False)
            prob = prob[0,0].cpu().numpy()
            mask = (prob >= thresh).astype(np.uint8)

            # Save Binary Prediction
            pred_dir = os.path.join(out_dir, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            plt.imsave(os.path.join(pred_dir, f"{z:03d}.png"), mask, cmap="gray")

            # Save Overlay (Optional)
            if overlay:
                gray = hu_to_gray(vol[z], overlay_window)
                rgb = np.stack([gray, gray, gray], axis=-1)
                rgb[...,0] = np.maximum(rgb[...,0], mask * overlay_alpha) 
                ov_dir = os.path.join(out_dir, "overlay")
                os.makedirs(ov_dir, exist_ok=True)
                plt.imsave(os.path.join(ov_dir, f"{z:03d}.png"), rgb)


# ========================= MULTIPROCESSING WORKER =========================
g_model = None
g_config = None

def worker_init(cfg_dict, ckpt_path, device):
    """
    Called once when a worker process starts. 
    Loads the model into the specific process's memory/GPU.
    """
    global g_model, g_config
    g_config = cfg_dict
    
    sys.stdout = open(os.devnull, 'w')
    try:
        g_model = build_siamese_model(cfg_dict, ckpt_path, device)
    finally:
        sys.stdout.close()
        sys.stdout = sys.__stdout__ 

def process_patient(row):
    """
    Worker function to process a single patient row from the CSV.
    """
    global g_model, g_config
    
    cfg = g_config
    model = g_model
    
    pid = (row.get("patient_id") or "").strip()
    if not pid: return None

    try:
        manifest = split_dicoms_to_series(
            patient_id=pid,
            study_date=row.get("study_date") or None,
            in_root=cfg["DICOM_ROOT"],
            out_root=cfg["WORK_ROOT"]
        )

        phase_filter = (row.get("phase") or "").lower()
        selected = []
        for m in manifest:
            if phase_filter and m["phase"] != phase_filter:
                continue
            selected.append(m)
        if not selected:
            selected = manifest

        for m in selected:
            series_dir = m["out_dir"]
            rel = os.path.relpath(series_dir, cfg["WORK_ROOT"])
            out_dir = os.path.join(cfg["OUT_ROOT"], rel)

            infer_and_save_pngs(
                model=model,
                dicom_dir=series_dir,
                out_dir=out_dir,
                phase=m["phase"],
                img_size=cfg["IMG_SIZE"],
                hu_windows=cfg["HU_WINDOWS"],
                z_context=cfg["Z_CONTEXT"],
                device=cfg["DEVICE"],
                thresh=cfg["THRESH"]
            )
        return pid 
    except Exception as e:
        return f"ERROR ({pid}): {str(e)}"

# ========================= MAIN =========================
if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    raw_cfg = load_config("config.yaml")
    

    config_payload = {
        "paths": raw_cfg.get("paths", {}), 
        "DICOM_ROOT": raw_cfg["paths"].get("dicom_root", "data/dicom"),
        "WORK_ROOT": raw_cfg["paths"].get("working_root", "data/working"),
        "OUT_ROOT": raw_cfg["paths"].get("output_preds_root", "outputs/pred_pngs"),
        "IMG_SIZE": (512,512),
        "Z_CONTEXT": 2,
        "THRESH": 0.5,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "HU_WINDOWS": [(-400,1600),(40,400),(-30,80),(-150,2000),(-1000,-300)],
        **raw_cfg 
    }

    INPUT_CSV = config_payload["paths"].get("patients_csv", "data/input.csv")
    CKPT_PATH = "outputs/runs/run_20251205_0117/ckpt/best_model.pth"
    COUNT_CSV = config_payload["paths"].get("counting_csv", "outputs/predicted/counting.csv")

    if not os.path.exists(INPUT_CSV):
        rows = [d for d in os.listdir(config_payload["DICOM_ROOT"]) 
                if os.path.isdir(os.path.join(config_payload["DICOM_ROOT"], d))]
        create_csv_from_patient_list(rows, INPUT_CSV)

    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))
    
    if not rows:
        raise RuntimeError("Input CSV is empty")

    print(f"[INFO] Total patients: {len(rows)}")
    print(f"[INFO] Starting Pool with {MAX_WORKERS} workers...")
    
    # Start the Pool
    with Pool(processes=MAX_WORKERS, 
              initializer=worker_init, 
              initargs=(config_payload, CKPT_PATH, config_payload["DEVICE"])) as pool:
        
        results = list(tqdm(pool.imap_unordered(process_patient, rows), total=len(rows)))

    # Check for any errors returned by workers
    errors = [r for r in results if r and str(r).startswith("ERROR")]
    if errors:
        print("\n[WARNING] Errors occurred during processing:")
        for e in errors: print(e)

    print("\n[COUNTING] Starting pixel counting...")
    counting(config_payload["WORK_ROOT"], config_payload["OUT_ROOT"], COUNT_CSV)
    print(f"[DONE] Counting results saved → {COUNT_CSV}")
    print("\nALL TASKS COMPLETED SUCCESSFULLY.\n")