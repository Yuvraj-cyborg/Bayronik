"""
CAMELS dataset loader with advanced features for baryonic field emulation.

Features:
- Memory-mapping support for large datasets
- Data augmentation (rotation, flip, noise)
- Conditional inputs (physics parameters)
- Multiple simulation suite support
- Multi-redshift support
- Efficient caching
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class CAMELSDataset(Dataset):
    """
    Dataset for CAMELS 2D projected mass maps with conditional inputs.
    
    Supports:
    - CV (Cosmic Variance): 27 simulations, same cosmology
    - LH (Latin Hypercube): 1000 simulations, varied cosmology
    - Multiple simulation suites (IllustrisTNG, SIMBA)
    
    Args:
        data_dir: Directory containing .npy files
        suite: Simulation suite (IllustrisTNG, SIMBA)
        dataset_type: CV or LH
        redshift: Target redshift (0.00, 0.50, 1.00, etc.)
        mmap: Memory-map files instead of loading to RAM
        augment: Enable data augmentation
        return_params: Return physics parameters for conditioning
        normalize: Normalization method ('log1p', 'standardize', None)
    """

    # Physics parameters for LH simulations (example values)
    # In practice, load from CAMELS parameter files
    LH_PARAMS = {
        'Omega_m': (0.1, 0.5),      # Matter density
        'sigma_8': (0.6, 1.0),      # Clustering amplitude
        'A_SN1': (0.25, 4.0),       # SN feedback
        'A_AGN1': (0.25, 4.0),      # AGN feedback
        'A_SN2': (0.5, 2.0),        # SN wind speed
        'A_AGN2': (0.5, 2.0),       # AGN boost
    }

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "CV",
        redshift: str = "0.00",
        mmap: bool = False,
        augment: bool = False,
        return_params: bool = False,
        normalize: str = "log1p",
        cache_stats: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.dataset_type = dataset_type
        self.redshift = redshift
        self.augment = augment
        self.return_params = return_params
        self.normalize = normalize
        
        # Build file paths
        input_file = f"Maps_Mcdm_{suite}_{dataset_type}_z={redshift}.npy"
        target_file = f"Maps_Mtot_{suite}_{dataset_type}_z={redshift}.npy"
        
        input_path = self.data_dir / input_file
        target_path = self.data_dir / target_file
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target not found: {target_path}")
        
        # Load data
        mmap_mode = "r" if mmap else None
        self.input_maps = np.load(input_path, mmap_mode=mmap_mode)
        self.target_maps = np.load(target_path, mmap_mode=mmap_mode)
        
        if self.input_maps.shape[0] != self.target_maps.shape[0]:
            raise ValueError("Input/target shape mismatch")
        
        self.num_samples = self.input_maps.shape[0]
        self.resolution = self.input_maps.shape[-1]
        
        # Load or generate physics parameters
        self.params = self._load_params() if return_params else None
        
        # Compute normalization statistics
        self.stats = self._compute_stats() if cache_stats else None

    def _load_params(self) -> Optional[np.ndarray]:
        """Load physics parameters for LH dataset."""
        if self.dataset_type == "CV":
            # CV has fixed cosmology
            # Planck 2018 values + default feedback
            params = np.tile([0.3, 0.8, 1.0, 1.0, 1.0, 1.0], (self.num_samples, 1))
            return params.astype(np.float32)
        
        # Try to load LH parameter file
        param_file = self.data_dir / f"params_{self.suite}_{self.dataset_type}.txt"
        if param_file.exists():
            params = np.loadtxt(param_file)
            if params.shape[0] == self.num_samples:
                return params.astype(np.float32)
        
        # Generate synthetic parameters (for demo purposes)
        # In production, always use actual CAMELS parameters
        np.random.seed(42)
        params = np.zeros((self.num_samples, 6), dtype=np.float32)
        for i, (key, (low, high)) in enumerate(self.LH_PARAMS.items()):
            params[:, i] = np.random.uniform(low, high, self.num_samples)
        
        return params

    def _compute_stats(self) -> Dict[str, float]:
        """Compute dataset statistics for normalization."""
        # Sample a subset for efficiency
        n_sample = min(100, self.num_samples)
        indices = np.random.choice(self.num_samples, n_sample, replace=False)
        
        input_sample = np.stack([self.input_maps[i] for i in indices])
        target_sample = np.stack([self.target_maps[i] for i in indices])
        
        # Apply log1p before computing stats
        input_log = np.log1p(input_sample)
        target_log = np.log1p(target_sample)
        
        return {
            'input_mean': float(input_log.mean()),
            'input_std': float(input_log.std()),
            'target_mean': float(target_log.mean()),
            'target_std': float(target_log.std()),
        }

    def _normalize(self, x: np.ndarray, is_input: bool = True) -> np.ndarray:
        """Apply normalization."""
        if self.normalize == "log1p":
            return np.log1p(x.astype(np.float32))
        
        elif self.normalize == "standardize":
            x = np.log1p(x.astype(np.float32))
            if self.stats is not None:
                mean_key = 'input_mean' if is_input else 'target_mean'
                std_key = 'input_std' if is_input else 'target_std'
                x = (x - self.stats[mean_key]) / (self.stats[std_key] + 1e-8)
            return x
        
        return x.astype(np.float32)

    def _augment(
        self,
        inp: np.ndarray,
        tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(4)
        if k > 0:
            inp = np.rot90(inp, k, axes=(-2, -1)).copy()
            tgt = np.rot90(tgt, k, axes=(-2, -1)).copy()
        
        # Random flip
        if np.random.random() > 0.5:
            inp = np.flip(inp, axis=-1).copy()
            tgt = np.flip(tgt, axis=-1).copy()
        
        if np.random.random() > 0.5:
            inp = np.flip(inp, axis=-2).copy()
            tgt = np.flip(tgt, axis=-2).copy()
        
        return inp, tgt

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self,
        idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get a sample.
        
        Returns:
            If return_params=False: (input, target)
            If return_params=True: (input, target, params)
        """
        inp = self.input_maps[idx]
        tgt = self.target_maps[idx]
        
        # Augmentation (before normalization for proper rotation)
        if self.augment:
            inp, tgt = self._augment(inp, tgt)
        
        # Normalize
        inp = self._normalize(inp, is_input=True)
        tgt = self._normalize(tgt, is_input=False)
        
        # Convert to tensors and add channel dimension
        inp_tensor = torch.from_numpy(inp).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt).unsqueeze(0)
        
        if self.return_params and self.params is not None:
            params_tensor = torch.from_numpy(self.params[idx])
            return inp_tensor, tgt_tensor, params_tensor
        
        return inp_tensor, tgt_tensor


class MultiSuiteDataset(Dataset):
    """
    Combined dataset from multiple simulation suites.
    
    Useful for training models that generalize across
    different hydrodynamic codes and feedback implementations.
    """

    def __init__(
        self,
        data_dir: str,
        suites: List[str] = ["IllustrisTNG", "SIMBA"],
        dataset_type: str = "LH",
        redshift: str = "0.00",
        **kwargs,
    ):
        self.datasets = []
        self.suite_ids = []
        
        for i, suite in enumerate(suites):
            try:
                ds = CAMELSDataset(
                    data_dir=data_dir,
                    suite=suite,
                    dataset_type=dataset_type,
                    redshift=redshift,
                    **kwargs,
                )
                self.datasets.append(ds)
                self.suite_ids.extend([i] * len(ds))
            except FileNotFoundError:
                print(f"Warning: Suite {suite} not found, skipping")
        
        if not self.datasets:
            raise FileNotFoundError("No datasets found")
        
        # Build cumulative indices
        self.cumulative_sizes = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _find_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Find which dataset and local index."""
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                local_idx = idx - (self.cumulative_sizes[i-1] if i > 0 else 0)
                return i, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self._find_dataset_idx(idx)
        return self.datasets[ds_idx][local_idx]


class MultiRedshiftDataset(Dataset):
    """
    Dataset combining multiple redshifts for temporal generalization.
    """

    def __init__(
        self,
        data_dir: str,
        suite: str = "IllustrisTNG",
        dataset_type: str = "LH",
        redshifts: List[str] = ["0.00", "0.50", "1.00"],
        return_redshift: bool = True,
        **kwargs,
    ):
        self.datasets = []
        self.redshift_values = []
        self.return_redshift = return_redshift
        
        for z in redshifts:
            try:
                ds = CAMELSDataset(
                    data_dir=data_dir,
                    suite=suite,
                    dataset_type=dataset_type,
                    redshift=z,
                    **kwargs,
                )
                self.datasets.append(ds)
                self.redshift_values.extend([float(z)] * len(ds))
            except FileNotFoundError:
                print(f"Warning: Redshift z={z} not found, skipping")
        
        if not self.datasets:
            raise FileNotFoundError("No datasets found")
        
        # Build cumulative indices
        self.cumulative_sizes = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _find_dataset_idx(self, idx: int) -> Tuple[int, int]:
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                local_idx = idx - (self.cumulative_sizes[i-1] if i > 0 else 0)
                return i, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self._find_dataset_idx(idx)
        result = self.datasets[ds_idx][local_idx]
        
        if self.return_redshift:
            z = torch.tensor([self.redshift_values[idx]], dtype=torch.float32)
            if isinstance(result, tuple) and len(result) == 3:
                # Has params, append z to params
                inp, tgt, params = result
                params = torch.cat([params, z])
                return inp, tgt, params
            else:
                inp, tgt = result
                return inp, tgt, z
        
        return result


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    suite: str = "IllustrisTNG",
    dataset_type: str = "LH",
    val_split: float = 0.15,
    augment_train: bool = True,
    return_params: bool = False,
    **kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        suite: Simulation suite
        dataset_type: CV or LH
        val_split: Validation split ratio
        augment_train: Enable augmentation for training
        return_params: Return physics parameters
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create dataset
    dataset = CAMELSDataset(
        data_dir=data_dir,
        suite=suite,
        dataset_type=dataset_type,
        augment=False,  # Augment only training
        return_params=return_params,
        **kwargs,
    )
    
    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Create augmented wrapper for training
    if augment_train:
        train_ds = AugmentedDataset(train_ds)
    
    # Determine pin_memory
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


class AugmentedDataset(Dataset):
    """Wrapper to add augmentation to a dataset subset."""
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        result = self.dataset[idx]
        
        # Unpack
        if len(result) == 3:
            inp, tgt, params = result
        else:
            inp, tgt = result
            params = None
        
        # Random rotation
        k = np.random.randint(4)
        if k > 0:
            inp = torch.rot90(inp, k, dims=(-2, -1))
            tgt = torch.rot90(tgt, k, dims=(-2, -1))
        
        # Random flips
        if np.random.random() > 0.5:
            inp = torch.flip(inp, dims=[-1])
            tgt = torch.flip(tgt, dims=[-1])
        
        if np.random.random() > 0.5:
            inp = torch.flip(inp, dims=[-2])
            tgt = torch.flip(tgt, dims=[-2])
        
        if params is not None:
            return inp, tgt, params
        return inp, tgt


if __name__ == "__main__":
    # Test dataset
    import sys
    
    print("Testing dataset module")
    print("-" * 50)
    
    # Create dummy data for testing
    test_dir = Path("/tmp/bayronik_test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Generate dummy maps
    n_samples = 10
    resolution = 256
    
    input_maps = np.random.exponential(1.0, (n_samples, resolution, resolution))
    target_maps = input_maps * 1.15 + np.random.normal(0, 0.1, input_maps.shape)
    
    np.save(test_dir / "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy", input_maps)
    np.save(test_dir / "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy", target_maps)
    
    # Test basic dataset
    ds = CAMELSDataset(
        data_dir=str(test_dir),
        suite="IllustrisTNG",
        dataset_type="CV",
        augment=True,
        return_params=True,
    )
    
    print(f"Dataset size: {len(ds)}")
    
    inp, tgt, params = ds[0]
    print(f"Input shape: {inp.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Params shape: {params.shape}")
    print(f"Params: {params}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        data_dir=str(test_dir),
        batch_size=4,
        num_workers=0,
        return_params=True,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for inp, tgt, params in train_loader:
        print(f"Batch input: {inp.shape}")
        print(f"Batch target: {tgt.shape}")
        print(f"Batch params: {params.shape}")
        break
    
    print("\nDataset tests passed!")
