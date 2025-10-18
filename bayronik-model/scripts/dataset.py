import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class CAMELSDataset(Dataset):
    """
    Custom PyTorch Dataset for loading CAMELS 2D projected mass maps
    from the newer CMD .npy file format.

    Args:
        root_dir (str): The directory containing the CAMELS data files.
        suite (str): The simulation suite to use (e.g., 'IllustrisTNG').
        dataset_type (str): The simulation set (e.g., 'CV', 'LH').
    """
    def __init__(self, root_dir, suite='IllustrisTNG', dataset_type='CV'):
        self.root_dir = root_dir

        # Construct the file paths for the required fields
        # Input is Dark Matter (Mcdm)
        input_filename = f"Maps_Mcdm_{suite}_{dataset_type}_z=0.00.npy"
        # Target is Total Matter (Mtot)
        target_filename = f"Maps_Mtot_{suite}_{dataset_type}_z=0.00.npy"

        input_path = os.path.join(root_dir, input_filename)
        target_path = os.path.join(root_dir, target_filename)

        print(f"Loading input maps from: {input_path}")
        print(f"Loading target maps from: {target_path}")

        if not os.path.exists(input_path) or not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Could not find required .npy files in {root_dir}. "
                "Please ensure both '{input_filename}' and '{target_filename}' are present."
            )

        # Load the entire dataset into memory.
        # The shape will be (num_simulations, height, width)
        self.input_maps = np.load(input_path).astype(np.float32)
        self.target_maps = np.load(target_path).astype(np.float32)

        if self.input_maps.shape[0] != self.target_maps.shape[0]:
            raise ValueError("Input and target files have a different number of maps!")

        print(f"Successfully loaded {self.input_maps.shape[0]} map pairs.")

    def __len__(self):
        return self.input_maps.shape[0]

    def __getitem__(self, idx):
        # Get the specific map from the pre-loaded arrays
        input_map = self.input_maps[idx]
        target_map = self.target_maps[idx]

        # Normalize the maps
        input_map = np.log1p(input_map)
        target_map = np.log1p(target_map)

        # Convert to PyTorch tensors and add a channel dimension
        input_tensor = torch.from_numpy(input_map).unsqueeze(0)
        target_tensor = torch.from_numpy(target_map).unsqueeze(0)

        return input_tensor, target_tensor


if __name__ == '__main__':
    # --- This is the updated test block ---
    print("\n--- Testing CAMELSDataset Loader with .npy files ---")

    test_data_dir = '../data/'
    print(f"Looking for data in: {os.path.abspath(test_data_dir)}")

    try:
        # 1. Instantiate the dataset using the 'CV' set for testing
        dataset = CAMELSDataset(root_dir=test_data_dir, dataset_type='CV')

        # 2. Get the first item
        input_tensor, target_tensor = dataset[0]
        print("\nSuccessfully loaded the first item.")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Target tensor shape: {target_tensor.shape}")
        assert input_tensor.shape == (1, 256, 256)

        # 3. Test with a DataLoader
        dataloader = DataLoader(dataset, batch_size=4) # Use a batch size of 4 for testing
        batch_input, batch_target = next(iter(dataloader))
        print("\nSuccessfully loaded a batch with DataLoader.")
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        assert batch_input.shape == (4, 1, 256, 256)

        print("\n--- Test PASSED ---")

    except Exception as e:
        print(f"\n--- Test FAILED ---")
        print(f"An error occurred: {e}")
        print("\nPlease check the following:")
        print("1. Did you download the TWO required files for the 'CV' set?")
        print("   - 'Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy'")
        print("   - 'Maps_Mtot_IllustrisTNG_CV_z=0.00.npy'")
        print("2. Did you place them inside the 'bayronik-model/data/' directory?")

