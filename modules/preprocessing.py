# modules/preprocessing.py
import os, glob, re, csv, shutil
from collections import defaultdict
from typing import Dict, List, Tuple
import pydicom

# -------------------------------
# Helpers
# -------------------------------
def safe_int(x, default=0):
    """
    Safely convert x to int; on failure return `default`.
    """
    try:
        return int(x)
    except Exception:
        return default

def sanitize(s: str, repl: str = "_"):
    """
    Make a string filesystem-friendly:
    - strip whitespace
    - replace non [A-Za-z0-9_.-] with `repl`
    - limit length to 80 characters to avoid excessively long paths
    """
    if not s:
        return "NA"
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", repl, s)
    return s[:80]

def guess_phase(series_desc: str):
    """
    Heuristically determine respiratory phase from SeriesDescription.
    Handles both full words and short tokens like INS/EXP.
    Returns one of: 'inhalation' | 'exhalation' | 'unknown'.
    """
    if not series_desc:
        return "unknown"
    sd = series_desc.lower()

    # Strong signals (whole-word matches)
    if re.search(r"\b(inspiration|inspiratory|inspir|inhal|ins)\b", sd):
        return "inhalation"
    if re.search(r"\b(expiration|expiratory|expir|exhal|exp)\b", sd):
        return "exhalation"

    # Tolerant patterns: _INS_, _EXP_, start/end tokens, etc.
    if "_ins" in sd or " ins " in sd or sd.endswith("_ins") or "ins_" in sd:
        return "inhalation"
    if "_exp" in sd or " exp " in sd or sd.endswith("_exp") or "exp_" in sd:
        return "exhalation"

    # Short markers
    if "insp" in sd:
        return "inhalation"
    if "exp" in sd:
        return "exhalation"

    return "unknown"

def collect_dicoms(patient_dir: str) -> List[str]:
    """
    Recursively collect all .dcm files under `patient_dir`.
    """
    files = glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True)
    return [f for f in files if os.path.isfile(f)]

def _get_date_from_ds(ds) -> str:
    """
    Extract YYYYMMDD from a DICOM dataset using the first available of:
    StudyDate > SeriesDate > AcquisitionDate > ContentDate.
    Falls back to 'unknown' if none found.
    """
    for tag in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
        val = getattr(ds, tag, None)
        if val:
            digits = re.sub(r"\D+", "", str(val))
            if len(digits) >= 8:
                return digits[:8]
    return "unknown"

def read_basic_tags_and_date(fp: str) -> Dict:
    """
    Read a minimal set of tags from a DICOM file without loading pixels.
    Returns a dict with keys:
      - date: 'YYYYMMDD' (or 'unknown')
      - series_number: int (or -1 on error)
      - series_desc: str (or 'ERR:<exception>' on error)
      - instance_number: int (or -1 on error)
    """
    try:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
        return {
            "date": _get_date_from_ds(ds),
            "series_number": safe_int(getattr(ds, "SeriesNumber", 0)),
            "series_desc": str(getattr(ds, "SeriesDescription", "") or ""),
            "instance_number": safe_int(getattr(ds, "InstanceNumber", 0)),
        }
    except Exception as e:
        return {
            "date": "unknown",
            "series_number": -1,
            "series_desc": f"ERR:{e}",
            "instance_number": -1,
        }

# -------------------------------
# Core
# -------------------------------
def split_dicoms_to_series(
    patient_id: str,
    study_date: str = None,
    in_root: str = "data/dicom",
    out_root: str = "data/working"
) -> List[Dict]:
    """
    Discover and re-organize all DICOM slices for one patient into clean series folders.

    Search scope:
      data/dicom/<patient_id>/**

    Behavior:
      1) For each found DICOM, read a minimal set of metadata.
      2) Determine an effective study date (YYYYMMDD) from metadata
         (StudyDate > SeriesDate > AcquisitionDate > ContentDate).
         If `study_date` is provided, only keep files whose derived date equals it.
      3) Group files by (date, SeriesNumber, SeriesDescription).
      4) For each group:
         - infer respiratory phase (inhalation/exhalation/unknown) from SeriesDescription
         - create output directory:
             data/working/<patient_id>/<YYYYMMDD>/Series<NNN>_<sanitized_desc>/
         - sort by InstanceNumber and copy slices as IM-00001.dcm, IM-00002.dcm, ...

    Returns:
      A manifest (list of dicts) with keys:
        patient_id, study_date, series_number, series_desc, phase,
        out_dir, num_slices, min_instance, max_instance

    Side effects:
      - Creates per-series output folders
      - Writes a CSV manifest at: data/working/<patient_id>/series_manifest.csv

    Raises:
      FileNotFoundError: if input patient directory is missing
      RuntimeError: if no DICOM files are found or none match `study_date`
    """
    patient_dir = os.path.join(in_root, patient_id)
    if not os.path.isdir(patient_dir):
        raise FileNotFoundError(f"Patient dir not found: {patient_dir}")

    files = collect_dicoms(patient_dir)
    if not files:
        raise RuntimeError(f"No DICOM found under {patient_dir}")

    # Group by (date, series_number, series_desc)
    groups: Dict[Tuple[str, int, str], List[Tuple[str, Dict]]] = defaultdict(list)
    for fp in files:
        tags = read_basic_tags_and_date(fp)
        d = tags["date"]
        if study_date and str(study_date).strip():
            if d != str(study_date):
                continue
        key = (d, tags["series_number"], tags["series_desc"])
        groups[key].append((fp, tags))

    if not groups:
        raise RuntimeError(
            f"No DICOM matched for patient_id={patient_id} "
            f"(study_date={study_date or 'AUTO from metadata'})"
        )

    all_manifest = []
    patient_out_base = os.path.join(out_root, patient_id)
    os.makedirs(patient_out_base, exist_ok=True)

    # Sort groups by (date, series_number)
    for (date_str, sn, desc), items in sorted(groups.items(), key=lambda k: (k[0][0], k[0][1])):
        phase = guess_phase(desc)
        date_label = date_str if date_str else "unknown"

        out_base = os.path.join(patient_out_base, date_label)
        os.makedirs(out_base, exist_ok=True)

        series_name = f"Series{sn:03d}_{sanitize(desc)}"
        out_dir = os.path.join(out_base, series_name)
        os.makedirs(out_dir, exist_ok=True)

        # Copy slices ordered by InstanceNumber
        items_sorted = sorted(items, key=lambda it: it[1]["instance_number"])
        for src_fp, t in items_sorted:
            dst_name = f"IM-{t['instance_number']:05d}.dcm"
            dst_fp = os.path.join(out_dir, dst_name)
            if not os.path.exists(dst_fp):
                shutil.copy2(src_fp, dst_fp)

        all_manifest.append({
            "patient_id": patient_id,
            "study_date": date_label,
            "series_number": sn,
            "series_desc": desc,
            "phase": phase,
            "out_dir": out_dir,
            "num_slices": len(items_sorted),
            "min_instance": items_sorted[0][1]["instance_number"] if items_sorted else "",
            "max_instance": items_sorted[-1][1]["instance_number"] if items_sorted else "",
        })

        print(
            f"[PREPROCESS] {patient_id} | {date_label} | {series_name}: "
            f"{len(items_sorted)} slices → {out_dir} [{phase}]"
        )

    # Write overall manifest CSV
    manifest_csv = os.path.join(patient_out_base, "series_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(all_manifest)
    print(f"[OK] Manifest saved → {manifest_csv}\n")

    return all_manifest
