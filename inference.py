import os, csv
import numpy as np
import torch
import pydicom
from tqdm import tqdm

from modules.preprocessing import split_dicoms_to_series
from modules.counting import counting
from modules.utils import load_config, create_csv_from_patient_list

# ========================= DICOM → HU =========================
def read_dicom_series(dicom_dir: str):
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

    return np.stack(vol, axis=0).astype(np.float32)  # [Z,H,W]


def apply_hu_windows(vol, windows):
    out = []
    for lo, hi in windows:
        ch = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
        out.append(ch)
    return np.stack(out, axis=1)  # [Z,C,H,W]


# ========================= SIAMESE MODEL LOADER =========================
from model.siamese_unet import SiameseUNetSCJAG

def build_siamese_model(cfg, ckpt_path, device):
    model = SiameseUNetSCJAG(config=cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        clean_state[k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"[MODEL] Loaded SiameseUNetSCJAG | missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device).eval()
    return model


# ========================= INFERENCE =========================
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
        for z in tqdm(range(Z), desc=f"Infer ({phase})"):
            ctx = []
            for zz in range(z - z_context, z + z_context + 1):
                zz = max(0, min(Z - 1, zz))
                ctx.append(chw[zz])

            x = np.concatenate(ctx, axis=0)[None]  # [1,C',H,W]
            x = torch.from_numpy(x).float().to(device)

            if list(x.shape[-2:]) != list(img_size):
                x = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)

            # -------- SIAMESE FORWARD --------
            out = model(x, x, alpha=0.0)
            logit = out["pred_mask_ex"] if phase == "ex" else out["pred_mask_in"]
            prob = torch.sigmoid(logit)

            prob = F.interpolate(prob, size=(H,W), mode="bilinear", align_corners=False)
            prob = prob[0,0].cpu().numpy()
            mask = (prob >= thresh).astype(np.uint8)

            pred_dir = os.path.join(out_dir, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            plt.imsave(os.path.join(pred_dir, f"{z:03d}.png"), mask, cmap="gray")

            if overlay:
                gray = hu_to_gray(vol[z], overlay_window)
                rgb = np.stack([gray, gray, gray], axis=-1)
                rgb[...,0] = np.maximum(rgb[...,0], mask * overlay_alpha)
                ov_dir = os.path.join(out_dir, "overlay")
                os.makedirs(ov_dir, exist_ok=True)
                plt.imsave(os.path.join(ov_dir, f"{z:03d}.png"), rgb)

    print(f"[DONE] Inference saved → {out_dir}")


# ========================= MAIN =========================
if __name__ == "__main__":
    cfg = load_config("config.yaml")

    INPUT_CSV  = cfg["paths"].get("patients_csv", "data/input.csv")
    DICOM_ROOT = cfg["paths"].get("dicom_root", "data/dicom")
    WORK_ROOT  = cfg["paths"].get("working_root", "data/working")
    OUT_ROOT   = cfg["paths"].get("output_preds_root", "outputs/pred_pngs")
    CKPT_PATH  = "outputs/runs/run_20251205_0117/ckpt/best_model.pth"
    COUNT_CSV  = cfg["paths"].get("counting_csv", "outputs/predicted/counting.csv")

    IMG_SIZE  = (512,512)
    Z_CONTEXT = 2
    THRESH   = 0.5
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    HU_WINDOWS = [(-400,1600),(40,400),(-30,80),(-150,2000),(-1000,-300)]

    # -------- LOAD MODEL ONCE --------
    model = build_siamese_model(cfg, CKPT_PATH, DEVICE)

    if not os.path.exists(INPUT_CSV):
        rows = [d for d in os.listdir(DICOM_ROOT) if os.path.isdir(os.path.join(DICOM_ROOT, d))]
        create_csv_from_patient_list(rows, INPUT_CSV)

    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Input CSV empty")

    for i, row in enumerate(rows, 1):
        pid = (row.get("patient_id") or "").strip()
        if not pid:
            continue

        print(f"\n========== {pid} ==========")
        sdate = row.get("study_date") or None
        phase_filter = (row.get("phase") or "").lower()

        manifest = split_dicoms_to_series(
            patient_id=pid,
            study_date=sdate,
            in_root=DICOM_ROOT,
            out_root=WORK_ROOT
        )

        selected = []
        for m in manifest:
            if phase_filter and m["phase"] != phase_filter:
                continue
            selected.append(m)
        if not selected:
            selected = manifest

        for m in selected:
            series_dir = m["out_dir"]
            rel = os.path.relpath(series_dir, WORK_ROOT)
            out_dir = os.path.join(OUT_ROOT, rel)

            print(f"[INFO] Series={os.path.basename(series_dir)} | phase={m['phase']}")
            infer_and_save_pngs(
                model=model,
                dicom_dir=series_dir,
                out_dir=out_dir,
                phase=m["phase"],
                img_size=IMG_SIZE,
                hu_windows=HU_WINDOWS,
                z_context=Z_CONTEXT,
                device=DEVICE,
                thresh=THRESH
            )

    print("\n[COUNTING] Start...")
    counting(WORK_ROOT, OUT_ROOT, COUNT_CSV)
    print(f"[DONE] Counting saved → {COUNT_CSV}")
    print("\nALL DONE SUCCESSFULLY.\n")
