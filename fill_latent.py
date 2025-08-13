#!/usr/bin/env python3
"""
Fill the `latent_path` column in an ADNI file manifest.

Example conversion
------------------
image : ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3_Br_20070319113435616_S13408_I45107.nii
latent: ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3_Br_20070319113435616_S13408_I45107_latent.npz

Usage
-----
$ python fill_latent_path.py --in  dataset.csv          # overwrite in place
$ python fill_latent_path.py --in  dataset.csv --out dataset_filled.csv
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

###############################################################################
# Regex that swaps the last ".nii" OR ".nii.gz" with "_latent.npz"
###############################################################################
_RE_EXT = re.compile(r"\.nii(?:\.gz)?$", re.IGNORECASE)


def make_latent_path(img_path: str) -> str:
    """
    Return a latent‑path string by replacing the final '.nii' or '.nii.gz'
    in *img_path* with '_latent.npz'.

    Raises
    ------
    ValueError
        If *img_path* doesn’t end with a recognised extension.
    """
    if not _RE_EXT.search(img_path):
        raise ValueError(f"Unrecognised image extension: {img_path!r}")
    return _RE_EXT.sub("_latent.npz", img_path)


def fill_latent(df: pd.DataFrame) -> pd.DataFrame:
    """Create / update the `latent_path` column only where it is blank."""
    # Ensure the column exists (creates full‑NaN column if missing).
    if "latent_path" not in df.columns:
        df["latent_path"] = pd.NA

    # “Missing” means NaN or an empty / whitespace‑only string.
    mask_missing = df["latent_path"].isna() | (df["latent_path"].astype(str).str.strip() == "")

    df.loc[mask_missing, "latent_path"] = (
        df.loc[mask_missing, "image_path"].apply(make_latent_path)
    )
    return df


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate latent_path from image_path.")
    p.add_argument("--in", dest="inp", required=True, help="Input CSV")
    p.add_argument("--out", dest="out", help="Output CSV (default: overwrite input)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    inp: Path = Path(args.inp)
    out: Path = Path(args.out) if args.out else inp

    # Read as strings so paths don’t get mangled.
    df = pd.read_csv(inp, dtype=str)
    if "image_path" not in df.columns:
        sys.exit("CSV must contain an 'image_path' column.")

    df = fill_latent(df)
    df.to_csv(out, index=False)
    print(f"Latent paths written → {out}")


if __name__ == "__main__":
    main()

