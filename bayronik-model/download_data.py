#!/usr/bin/env python3
"""Download CAMELS data from Flatiron Institute."""

import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


BASE_URL = "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/IllustrisTNG"

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
