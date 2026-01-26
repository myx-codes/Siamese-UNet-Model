# modules/dataset.py
import os
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pydicom
import nibabel as nib

# =========================================================
# Helper: Mask Path Resolver (no dependency on Utils)
# =========================================================
def resolve_mask_path(mask_root: str, pid: str, phase: str) -> str:
    """
    Searches for the mask file in several possible locations:
    1. mask/{pid}/{phase}/mask.nii.gz
    2. mask/{pid}/{phase}.nii.gz
    3. mask/{pid}/{phase}.nii
    """
    # Variant 1: inside a subfolder (mask.nii.gz)
    p1 = os.path.join(mask_root, pid, phase, "mask.nii.gz")
    if os.path.exists(p1):
        return p1
    
    # Variant 2: direct file ({phase}.nii.gz)
    p2 = os.path.join(mask_root, pid, f"{phase}.nii.gz")
    if os.path.exists(p2):
        return p2

    # Variant 3: .nii extension
    p3 = os.path.join(mask_root, pid, f"{phase}.nii")
    if os.path.exists(p3):
        return p3
    
    return p1  # Return default path (will raise error later if not found)


# =========================================================
# Low-level I/O helpers
# =========================================================
def _read_dicom_series(dicom_dir: str) -> np.ndarray:
    """
    Reads a DICOM series and converts it to Hounsfield Units (HU).
    Output: [Z, H, W] float32
    """
    files = [f for f in glob.glob(os.path.join(dicom_dir, "*.dcm")) if os.path.isfile(f)]
    if not files:
        return np.zeros((0, 1, 1), dtype=np.float32)

    def _key(fp):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0

    files = sorted(files, key=_key)
    vol = []
    slope, inter = None, None
    for fp in files:
        ds = pydicom.dcmread(fp, force=True)
        px = ds.pixel_array.astype(np.int16)
        if slope is None:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = px * slope + inter
        vol.append(hu)
    vol = np.stack(vol, axis=0)  # [Z, H, W]
    return vol.astype(np.float32)


def load_and_transform_mask(path: str, phase: str) -> np.ndarray:
    """
    Loads a NIfTI mask.
    Important: Flip and rotation logic preserved for your specific dataset.
    """
    m = nib.load(path).get_fdata()
    m = np.transpose(m, (2, 0, 1))  # [H, W, Z] -> [Z, H, W]
    
    if phase in {'in', 'ex'}:
        # Specific orientation correction
        m = np.flip(m, axis=2)
        m = np.rot90(m, k=1, axes=(1, 2))
        
    return (m > 0).astype(np.uint8)


def _window_slice(x: np.ndarray, wl: float, ww: float) -> np.ndarray:
    """HU windowing and normalization to [0, 1]."""
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return x.astype(np.float32)


# =========================================================
# Shared mid-level helpers
# =========================================================
def _stack_z_context(
    vol_hu: np.ndarray,
    z: int,
    hu_windows: List[Tuple[float, float]],
    z_context: int
) -> np.ndarray:
    """
    Combines multiple HU windows and adjacent slices.
    Output: [Channels, H, W]
    """
    def get_chans(slice_data):
        chans = []
        for (lo, hi) in hu_windows:
            wl = 0.5 * (lo + hi)
            ww = (hi - lo)
            chans.append(_window_slice(slice_data, wl, ww))
        return np.stack(chans, axis=0)

    # Center slice
    final_stack = [get_chans(vol_hu[z])]
    
    # Adjacent slices (if z_context > 0)
    Z = vol_hu.shape[0]
    for dz in range(1, z_context + 1):
        # Lower neighbor
        z_prev = max(0, z - dz)
        final_stack.insert(0, get_chans(vol_hu[z_prev]))
        # Upper neighbor
        z_next = min(Z - 1, z + dz)
        final_stack.append(get_chans(vol_hu[z_next]))
        
    return np.concatenate(final_stack, axis=0)  # [C*(2*zc+1), H, W]


def _resize_xy(x_np: np.ndarray, y_np: np.ndarray, out_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image and mask to target size."""
    H, W = x_np.shape[-2], x_np.shape[-1]
    th, tw = out_hw
    if (H == th) and (W == tw):
        return x_np, y_np
    
    xt = torch.from_numpy(x_np[None, ...])  # [1, C, H, W]
    yt = torch.from_numpy(y_np[None, ...])  # [1, 1, H, W]
    
    xt = F.interpolate(xt, size=(th, tw), mode="bilinear", align_corners=False)
    yt = F.interpolate(yt, size=(th, tw), mode="nearest")  # Nearest for masks!
    
    return xt[0].numpy(), yt[0].numpy()


# =========================================================
# Paired IN–EX dataset (Siamese) - MAIN CLASS
# =========================================================
class CT2DDatasetPairs(Dataset):
    """
    Dataset for Siamese models.
    Each __getitem__ call returns an (Inhale, Exhale) pair.
    """
    def __init__(self,
                 patient_ids: List[str],
                 dicom_root: str,
                 mask_root: str,
                 img_size: Tuple[int, int] = (512, 512),
                 hu_windows: List[Tuple[float, float]] = ((-400, 1600), (40, 400), (-30, 80)),
                 z_context: int = 0,
                 pair_mode: str = "same_z",
                 augment: bool = False,
                 verbose: bool = False,
                 with_domain: bool = False):
        super().__init__()
        self.patient_ids = list(patient_ids)
        self.dicom_root = dicom_root
        self.mask_root = mask_root
        self.img_size = tuple(img_size)
        self.hu_windows = list(map(tuple, hu_windows))
        self.zc = int(z_context)
        self.augment = bool(augment)
        self.with_domain = bool(with_domain)

        self._vol_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._msk_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._index: List[Tuple[str, int]] = []  # (pid, z)

        # 1. Load and cache data
        for pid in self.patient_ids:
            ddir_in = os.path.join(self.dicom_root, pid, "in")
            ddir_ex = os.path.join(self.dicom_root, pid, "ex")
            mpath_in = resolve_mask_path(self.mask_root, pid, "in")
            mpath_ex = resolve_mask_path(self.mask_root, pid, "ex")

            if (not os.path.isdir(ddir_in)) or (not os.path.isdir(ddir_ex)):
                continue
            if (not os.path.exists(mpath_in)) or (not os.path.exists(mpath_ex)):
                continue

            vol_in = _read_dicom_series(ddir_in)
            vol_ex = _read_dicom_series(ddir_ex)

            if vol_in.shape[0] == 0 or vol_ex.shape[0] == 0:
                continue

            try:
                m_in = load_and_transform_mask(mpath_in, "in")
                m_ex = load_and_transform_mask(mpath_ex, "ex")
            except Exception as e:
                print(f"Error loading mask for {pid}: {e}")
                continue

            # Crop to the smallest Z dimension (in case slice counts differ)
            Z = min(vol_in.shape[0], vol_ex.shape[0], m_in.shape[0], m_ex.shape[0])
            
            self._vol_cache[(pid, "in")] = vol_in
            self._vol_cache[(pid, "ex")] = vol_ex
            self._msk_cache[(pid, "in")] = m_in
            self._msk_cache[(pid, "ex")] = m_ex

            # Build index
            for z in range(Z):
                self._index.append((pid, z))

        if verbose:
            print(f"[Dataset] Total pairs: {len(self._index)} | Patients: {len(self.patient_ids)}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        pid, z = self._index[idx]
        
        # Retrieve cached data
        vol_in = self._vol_cache[(pid, "in")]
        vol_ex = self._vol_cache[(pid, "ex")]
        m_in = self._msk_cache[(pid, "in")]
        m_ex = self._msk_cache[(pid, "ex")]

        # --- Inhale ---
        xin_np = _stack_z_context(vol_in, z, self.hu_windows, self.zc)
        yin_np = m_in[z].astype(np.float32)[None, ...]
        xin_np, yin_np = _resize_xy(xin_np, yin_np, self.img_size)

        # --- Exhale ---
        xex_np = _stack_z_context(vol_ex, z, self.hu_windows, self.zc)
        yex_np = m_ex[z].astype(np.float32)[None, ...]
        xex_np, yex_np = _resize_xy(xex_np, yex_np, self.img_size)

        # --- Augmentation (same for both branches in Siamese) ---
        if self.augment:
            # Horizontal Flip
            if np.random.rand() < 0.5:
                xin_np = xin_np[..., ::-1].copy()
                yin_np = yin_np[..., ::-1].copy()
                xex_np = xex_np[..., ::-1].copy()
                yex_np = yex_np[..., ::-1].copy()
            
            # Vertical Flip
            if np.random.rand() < 0.5:
                xin_np = xin_np[..., ::-1, :].copy()
                yin_np = yin_np[..., ::-1, :].copy()
                xex_np = xex_np[..., ::-1, :].copy()
                yex_np = yex_np[..., ::-1, :].copy()

        # Convert to tensors
        xin = torch.from_numpy(xin_np)
        yin = torch.from_numpy(yin_np)
        xex = torch.from_numpy(xex_np)
        yex = torch.from_numpy(yex_np)
        
        # Metadata (if needed)
        meta = (pid, torch.tensor([z], dtype=torch.int32))

        # For domain-adversarial training (0=IN, 1=EX)
        if self.with_domain:
            domain_pair = torch.tensor([0.0, 1.0], dtype=torch.float32)
            return xin, yin, xex, yex, meta, domain_pair
        
        # Standard return (expected by training loop)
        return xin, yin, xex, yex, meta