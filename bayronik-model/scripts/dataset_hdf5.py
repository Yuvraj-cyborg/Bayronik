import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class CAMELSDatasetHDF5(Dataset):
    """
    PyTorch Dataset for loading CAMELS 2D projected mass maps from HDF5 files.
    
    HDF5 format is preferred because:
    - Single file contains all simulations (not one .npy per sim)
    - Compressed (5-10× smaller)
    - Includes metadata (parameters, redshifts, units)
    - Faster random access
    
    Args:
        root_dir (str): Directory containing HDF5 files
        suite (str): Simulation suite ('IllustrisTNG', 'SIMBA', etc.)
        dataset_type (str): Dataset type ('CV', 'LH', etc.)
        redshift (float): Which redshift slice to load (default: 0.0)
        cache_in_memory (bool): Load entire dataset into RAM (default: False)
    """
    def __init__(
        self, 
        root_dir, 
        suite='IllustrisTNG', 
        dataset_type='CV',
        redshift=0.0,
        cache_in_memory=False
    ):
        self.root_dir = root_dir
        self.suite = suite
        self.dataset_type = dataset_type
        self.redshift = redshift
        self.cache_in_memory = cache_in_memory
        
        # Construct file paths
        input_filename = f"Maps_Mcdm_{suite}_{dataset_type}_z={redshift:.2f}.hdf5"
        target_filename = f"Maps_Mtot_{suite}_{dataset_type}_z={redshift:.2f}.hdf5"
        
        self.input_path = os.path.join(root_dir, input_filename)
        self.target_path = os.path.join(root_dir, target_filename)
        
        print(f"Loading HDF5 dataset:")
        print(f"  Input:  {self.input_path}")
        print(f"  Target: {self.target_path}")
        
        # Check if files exist
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(
                f"Input HDF5 file not found: {self.input_path}\n"
                f"Please download from CAMELS: https://camels.readthedocs.io/"
            )
        if not os.path.exists(self.target_path):
            raise FileNotFoundError(
                f"Target HDF5 file not found: {self.target_path}"
            )
        
        # Open HDF5 files to get dataset info
        with h5py.File(self.input_path, 'r') as f:
            # HDF5 structure: {'Maps_Mcdm': (n_sims, height, width)}
            self.n_simulations = f['Maps_Mcdm'].shape[0]
            self.map_shape = f['Maps_Mcdm'].shape[1:]
            print(f"  Found {self.n_simulations} simulations, each {self.map_shape}")
        
        # Optionally cache entire dataset in memory
        if cache_in_memory:
            print("  Caching dataset in memory (this may take a moment)...")
            with h5py.File(self.input_path, 'r') as f:
                self.input_cache = f['Maps_Mcdm'][:].astype(np.float32)
            with h5py.File(self.target_path, 'r') as f:
                self.target_cache = f['Maps_Mtot'][:].astype(np.float32)
            print("  ✓ Dataset cached!")
        else:
            self.input_cache = None
            self.target_cache = None
            
    def __len__(self):
        return self.n_simulations
    
    def __getitem__(self, idx):
        """
        Returns:
            input_tensor: (1, H, W) - Dark matter map
            target_tensor: (1, H, W) - Total matter map
        """
        if self.cache_in_memory:
            # Use cached data
            input_map = self.input_cache[idx]
            target_map = self.target_cache[idx]
        else:
            # Load from disk on-the-fly
            with h5py.File(self.input_path, 'r') as f:
                input_map = f['Maps_Mcdm'][idx].astype(np.float32)
            with h5py.File(self.target_path, 'r') as f:
                target_map = f['Maps_Mtot'][idx].astype(np.float32)
        
        # Normalize: log1p handles large dynamic range
        input_map = np.log1p(input_map)
        target_map = np.log1p(target_map)
        
        # Convert to PyTorch tensors and add channel dimension
        input_tensor = torch.from_numpy(input_map).unsqueeze(0)
        target_tensor = torch.from_numpy(target_map).unsqueeze(0)
        
        return input_tensor, target_tensor


def load_camels_parameters(root_dir, suite='IllustrisTNG', dataset_type='CV'):
    """
    Load cosmology and baryonic feedback parameters for each simulation.
    
    CAMELS provides these in a text file: params_<suite>_<type>.txt
    
    Returns:
        params_df: pandas DataFrame with columns like:
            - Omega_m, sigma_8, A_SN1, A_SN2, A_AGN1, A_AGN2, etc.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. Cannot load parameters.")
        return None
    
    param_file = os.path.join(root_dir, f"params_{suite}_{dataset_type}.txt")
    
    if not os.path.exists(param_file):
        print(f"Warning: Parameter file not found: {param_file}")
        print("Simulations will not have associated parameters.")
        return None
    
    # Load parameter file (usually space-separated)
    params_df = pd.read_csv(param_file, delim_whitespace=True)
    print(f"Loaded {len(params_df)} parameter sets from {param_file}")
    
    return params_df


if __name__ == '__main__':
    """Test the HDF5 data loader"""
    print("\n=== Testing CAMELS HDF5 Dataset Loader ===\n")
    
    test_data_dir = '../data/'
    
    try:
        # 1. Test HDF5 loader (if files exist)
        print("Test 1: Loading HDF5 dataset...")
        dataset_hdf5 = CAMELSDatasetHDF5(
            root_dir=test_data_dir,
            suite='IllustrisTNG',
            dataset_type='CV',
            cache_in_memory=True  # Cache for speed
        )
        
        # Get first sample
        input_tensor, target_tensor = dataset_hdf5[0]
        print(f"\n✓ Successfully loaded first sample:")
        print(f"  Input shape:  {input_tensor.shape}")
        print(f"  Target shape: {target_tensor.shape}")
        
        # 2. Test parameter loading
        print("\nTest 2: Loading simulation parameters...")
        params = load_camels_parameters(test_data_dir)
        if params is not None:
            print(f"  Columns: {list(params.columns)}")
            print(f"  First row:\n{params.iloc[0]}")
        
        print("\n=== All Tests Passed! ===\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ HDF5 files not found: {e}")
        print("\nTo get HDF5 files:")
        print("1. Go to: https://camels.readthedocs.io/en/latest/data_access.html")
        print("2. Request access to the dataset")
        print("3. Download Maps_Mcdm_IllustrisTNG_CV_z=0.00.hdf5")
        print("4. Download Maps_Mtot_IllustrisTNG_CV_z=0.00.hdf5")
        print("5. Place them in bayronik-model/data/\n")
        
        print("Current directory contents:")
        print(os.listdir(test_data_dir))

