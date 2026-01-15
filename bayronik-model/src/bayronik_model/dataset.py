"""CAMELS dataset loader with memory-mapping support."""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CAMELSDataset(Dataset):
    """
    Dataset for CAMELS 2D projected mass maps.
    
    Args:
        data_dir: Directory containing .npy files
        suite: Simulation suite (default: IllustrisTNG)
        dataset_type: CV (27 sims) or LH (1000 sims)
        mmap: Memory-map files instead of loading to RAM
    """

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "CV",
        mmap: bool = False,
    ):
        self.data_dir = Path(data_dir)
        
        input_file = f"Maps_Mcdm_{suite}_{dataset_type}_z=0.00.npy"
        target_file = f"Maps_Mtot_{suite}_{dataset_type}_z=0.00.npy"
        
        input_path = self.data_dir / input_file
        target_path = self.data_dir / target_file

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target_path}")

        mmap_mode = "r" if mmap else None
        self.input_maps = np.load(input_path, mmap_mode=mmap_mode)
        self.target_maps = np.load(target_path, mmap_mode=mmap_mode)

        if self.input_maps.shape[0] != self.target_maps.shape[0]:
            raise ValueError("Input/target shape mismatch")

        self.num_samples = self.input_maps.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = np.log1p(self.input_maps[idx].astype(np.float32))
        tgt = np.log1p(self.target_maps[idx].astype(np.float32))
        
        return (
            torch.from_numpy(inp).unsqueeze(0),
            torch.from_numpy(tgt).unsqueeze(0),
        )
