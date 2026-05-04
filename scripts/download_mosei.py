#!/usr/bin/env python
"""Download the MultiBench-processed MOSEI pickle from Google Drive.

The MultiBench repo hosts processed affect datasets on Google Drive:
https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv

This script downloads mosei_senti_data.pkl (the packed pickle used by
MultiBench's affect pipeline) and renames it to mosei_raw.pkl for
consistency with our loader.

Usage:
    pip install gdown
    python scripts/download_mosei.py [--output data/mosei_raw.pkl]
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/mosei_raw.pkl",
        help="Where to save the downloaded pickle.",
    )
    parser.add_argument(
        "--folder-id",
        default="1A_hTmifi824gypelGobgl2M-5Rw9VWHv",
        help="Google Drive folder ID from MultiBench README.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip download if the output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)

    if args.skip_if_exists and output.exists():
        print(f"File already exists: {output}")
        return

    try:
        import gdown
    except ImportError:
        print("gdown is required. Install with: pip install gdown")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Download the entire MOSEI folder from MultiBench Google Drive.
    # The folder contains mosei_senti_data.pkl (and possibly mosei_raw.pkl).
    folder_url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    tmp_dir = output.parent / "_mosei_download_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MOSEI data from {folder_url} ...")
    gdown.download_folder(url=folder_url, output=str(tmp_dir), quiet=False)

    # Find the downloaded pickle file.
    candidates = list(tmp_dir.rglob("*.pkl"))
    if not candidates:
        print(f"ERROR: No .pkl file found in downloaded folder: {tmp_dir}")
        sys.exit(1)

    # Prefer mosei_raw.pkl, fall back to mosei_senti_data.pkl, else first .pkl.
    chosen = None
    for name in ("mosei_raw.pkl", "mosei_senti_data.pkl"):
        for c in candidates:
            if c.name == name:
                chosen = c
                break
        if chosen:
            break
    if chosen is None:
        chosen = candidates[0]

    chosen.rename(output)
    print(f"Saved MOSEI data to: {output}")

    # Quick sanity check.
    _validate_pickle(output)

    # Clean up temp directory.
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _validate_pickle(path: Path) -> None:
    """Quick check that the pickle has the expected MultiBench structure."""
    print(f"Validating {path} ...")
    with path.open("rb") as f:
        data = pickle.load(f)

    for split in ("train", "valid", "test"):
        if split not in data:
            print(f"  WARNING: missing split '{split}'")
            continue
        d = data[split]
        keys = set(d.keys())
        print(f"  {split}: keys={sorted(keys)}")
        for modality in ("text", "audio", "vision", "labels"):
            if modality in d:
                arr = d[modality]
                print(f"    {modality}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                print(f"    {modality}: MISSING")
    print("Validation complete.")


if __name__ == "__main__":
    main()
