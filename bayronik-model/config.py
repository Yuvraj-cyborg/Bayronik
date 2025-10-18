import torch
import os

# --- Machine Learning Configuration ---
# Set the device to use for training.
# Use MPS for Apple Silicon, CUDA for NVIDIA, otherwise CPU.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 15 # How many times to loop over the entire dataset

# --- THE FIX: Set NUM_WORKERS to 0 for stability on macOS ---
NUM_WORKERS = 0 # Number of CPU cores for data loading. 0 is safest.

# --- Dataset Configuration ---
# Define which CAMELS simulation suite and dataset type to use.
SUITE = "IllustrisTNG"
DATASET_TYPE = "CV" # Using the small "Cosmic Variance" set for local testing

# --- Directory and File Paths ---
# This makes the script portable by referencing file locations relative
# to the script's own location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_NAME = f"unet_{SUITE.lower()}_{DATASET_TYPE.lower()}_v1.pth"

