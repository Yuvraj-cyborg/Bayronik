import torch
import os

# Import our custom modules
from scripts.model import UNet
import config

def export_model_to_torchscript():
    """
    Loads the trained PyTorch model, converts it to TorchScript,
    and saves it for use in other environments like Rust.
    """
    print("--- Starting Model Export to TorchScript ---")

    # --- 1. Define Paths ---
    weights_path = os.path.join(config.WEIGHTS_DIR, config.MODEL_NAME)
    export_path = os.path.join(config.WEIGHTS_DIR, "traced_unet_model.pt")

    # --- 2. Check if the trained weights file exists ---
    if not os.path.exists(weights_path):
        print(f"Error: Trained weights file not found at '{weights_path}'")
        print("Please run train.py to train and save the model first.")
        return

    print(f"Loading trained weights from: {weights_path}")

    # --- 3. Initialize Model and Load Weights ---
    # We must use the same architecture as when we trained
    model = UNet(n_channels=1, n_classes=1)
    
    # Load the state dictionary (the learned weights)
    # We use map_location='cpu' to ensure it loads correctly on any machine
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode (important for tracing)
    model.eval()
    print("Model architecture created and weights loaded successfully.")

    # --- 4. Trace the Model ---
    # To convert to TorchScript, we "trace" it by passing a dummy input
    # through the model and recording the operations.
    dummy_input = torch.randn(1, 1, 256, 256)
    print("Tracing model with dummy input...")
    traced_script_module = torch.jit.trace(model, dummy_input)
    print("Model traced successfully.")

    # --- 5. Save the Traced Model ---
    traced_script_module.save(export_path)
    print(f"\n--- Export SUCCESSFUL ---")
    print(f"TorchScript model saved to: {export_path}")

if __name__ == "__main__":
    export_model_to_torchscript()

