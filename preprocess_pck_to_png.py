"""
Preprocess KneeMRI pickle (.pck) dataset into the image-folder structure
expected by the BYOL-ACL-Detection training pipeline.

Input  (Kaggle/Zenodo):
    data_input/
    ├── metadata.csv          # columns include filename and aclDiagnosis
    └── volumetric_data/      # (or root level) .pck files
        ├── 0001.pck
        ├── 0002.pck
        └── ...

Output (what the repo needs):
    data_output/
    ├── 0001_0/
    │   └── slice_00.png
    ├── 0002_1/
    │   └── slice_00.png
    └── ...

Usage on Kaggle:
    python preprocess_pck_to_png.py \
        --input_dir /kaggle/input/mri-knee \
        --output_dir /kaggle/working/processed_data \
        --slice_mode middle

Usage locally:
    python preprocess_pck_to_png.py \
        --input_dir ./raw_data \
        --output_dir ./processed_data \
        --slice_mode middle
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────────────

def find_metadata_csv(input_dir):
    """Search for metadata.csv anywhere under input_dir."""
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower() == "metadata.csv":
                return os.path.join(root, f)
    return None


def find_pck_files(input_dir):
    """Recursively find all .pck files under input_dir."""
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.pck"), recursive=True))


def load_pickle_volume(pck_path):
    """Load a pickle file and return the numpy array inside."""
    with open(pck_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")  # encoding for Python 2 pickles

    # The pickle may contain a raw ndarray or a dict with a 'data'/'volume' key.
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, dict):
        for key in ("data", "volume", "vol", "image", "img"):
            if key in data:
                return np.array(data[key])
        # If none of the known keys, return the first array-like value
        for v in data.values():
            arr = np.asarray(v)
            if arr.ndim >= 2:
                return arr
    if isinstance(data, (list, tuple)):
        return np.array(data)

    raise ValueError(f"Cannot interpret pickle contents of {pck_path} (type={type(data)})")


def normalize_to_uint8(arr):
    """Normalize a 2-D array to 0–255 uint8 for PNG export."""
    arr = arr.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 0:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def select_slices(volume, mode="middle"):
    """
    Given a 3-D volume (slices, H, W), return a list of (slice_index, 2d_array)
    according to the chosen mode.

    Modes:
        middle  – single middle slice (default, matches repo's slice_00.png)
        all     – every slice in the volume
    """
    if volume.ndim == 2:
        return [(0, volume)]

    num_slices = volume.shape[0]

    if mode == "middle":
        mid = num_slices // 2
        return [(0, volume[mid])]
    elif mode == "all":
        return [(i, volume[i]) for i in range(num_slices)]
    else:
        raise ValueError(f"Unknown slice_mode: {mode}")


# ── label resolution ─────────────────────────────────────────────────────────

def build_label_map(metadata_csv_path):
    """
    Parse metadata.csv and return a dict: stem (no extension) → class_label (int).

    The Zenodo/Kaggle KneeMRI metadata uses 1-indexed labels in a column called
    'aclDiagnosis' (1=healthy, 2=partial, 3=complete).  The repo uses 0-indexed
    labels, so we subtract 1.

    If the column name differs, we try several common alternatives.
    """
    df = pd.read_csv(metadata_csv_path)

    # ── locate the label column ──
    label_col = None
    for candidate in ("aclDiagnosis", "acl_diagnosis", "label", "Label",
                       "diagnosis", "Diagnosis", "class", "Class", "target"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print("[!] Could not auto-detect label column. Available columns:")
        print("   ", list(df.columns))
        sys.exit(1)

    # ── locate the filename / id column ──
    id_col = None
    for candidate in ("volumeFilename", "filename", "Filename", "file",
                       "File", "id", "ID", "name", "Name"):
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        print("[!] Could not auto-detect filename/id column. Available columns:")
        print("   ", list(df.columns))
        sys.exit(1)

    print(f"[*] Using id column = '{id_col}', label column = '{label_col}'")

    label_map = {}
    for _, row in df.iterrows():
        raw_id = str(row[id_col])
        stem = os.path.splitext(os.path.basename(raw_id))[0]  # strip path & extension
        raw_label = int(row[label_col])

        # Convert 1-indexed (Zenodo) → 0-indexed (repo) if labels start at 1
        label_map[stem] = raw_label

    # Auto-detect whether labels are 1-indexed and adjust
    unique_labels = sorted(set(label_map.values()))
    if min(unique_labels) == 1:
        print(f"[*] Detected 1-indexed labels {unique_labels}, converting to 0-indexed")
        label_map = {k: v - 1 for k, v in label_map.items()}
    else:
        print(f"[*] Labels are already 0-indexed: {unique_labels}")

    return label_map


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert KneeMRI .pck dataset to image-folder format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root of the raw Kaggle/Zenodo dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write the processed image folders")
    parser.add_argument("--slice_mode", type=str, default="middle",
                        choices=["middle", "all"],
                        help="Which slice(s) to extract from each volume "
                             "(default: middle)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Find metadata.csv
    csv_path = find_metadata_csv(input_dir)
    if csv_path is None:
        print("[!] metadata.csv not found anywhere under", input_dir)
        sys.exit(1)
    print(f"[*] Found metadata at: {csv_path}")

    label_map = build_label_map(csv_path)
    print(f"[*] Loaded labels for {len(label_map)} volumes")

    # 2. Find all .pck files
    pck_files = find_pck_files(input_dir)
    if not pck_files:
        print("[!] No .pck files found under", input_dir)
        sys.exit(1)
    print(f"[*] Found {len(pck_files)} pickle files")

    # 3. Process each pickle
    success, skipped, errors = 0, 0, 0
    class_counts = {}

    for pck_path in pck_files:
        stem = os.path.splitext(os.path.basename(pck_path))[0]

        if stem not in label_map:
            print(f"  [SKIP] {stem} — not in metadata.csv")
            skipped += 1
            continue

        label = label_map[stem]

        try:
            volume = load_pickle_volume(pck_path)
        except Exception as e:
            print(f"  [ERR]  {stem} — {e}")
            errors += 1
            continue

        slices = select_slices(volume, mode=args.slice_mode)

        for slice_idx, slice_2d in slices:
            if args.slice_mode == "all":
                folder_name = f"{stem}_slice{slice_idx:02d}_{label}"
            else:
                folder_name = f"{stem}_{label}"

            folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            img_arr = normalize_to_uint8(slice_2d)
            img = Image.fromarray(img_arr, mode="L")
            img.save(os.path.join(folder_path, "slice_00.png"))

        class_counts[label] = class_counts.get(label, 0) + 1
        success += 1

    # 4. Summary
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Processed : {success}")
    print(f"  Skipped   : {skipped}")
    print(f"  Errors    : {errors}")
    print(f"  Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"  Output at : {os.path.abspath(output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
