#!/usr/bin/env python3
"""Download CAMELS data from Flatiron Institute."""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm


BASE_URL = "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG"

CAMELS_PARAMS_RAW = (
    "https://raw.githubusercontent.com/franciscovillaescusa/CAMELS/master"
    "/docs/params/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt"
)

FILES = {
    "CV": [
        "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy",
        "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy",
    ],
    "LH": [
        "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy",
        "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy",
    ],
}


class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(dataset: str, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in FILES[dataset]:
        dest = data_dir / filename
        if dest.exists():
            print(f"Skip: {filename}")
            continue
        
        url = f"{BASE_URL}/{filename}"
        print(f"Downloading: {filename}")
        
        with ProgressBar(unit="B", unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(url, dest, reporthook=pbar.update_to)
    
    if dataset == "LH":
        params_dest = data_dir / "params_LH_IllustrisTNG.txt"
        if not params_dest.exists():
            raw_file = data_dir / "_CosmoAstroSeed_LH_raw.txt"
            print("Downloading LH params from CAMELS GitHub...")
            urllib.request.urlretrieve(CAMELS_PARAMS_RAW, raw_file)
            data = np.genfromtxt(raw_file, dtype=str, comments="#")
            params = data[:, 1:7].astype(np.float32)
            np.savetxt(params_dest, params, fmt="%.5f")
            raw_file.unlink()
            print(f"Saved {params.shape[0]} x {params.shape[1]} params to {params_dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CV", choices=["CV", "LH", "all"])
    parser.add_argument("--data-dir", type=str, default="data")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    datasets = ["CV", "LH"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        download(ds, data_dir)


if __name__ == "__main__":
    main()
