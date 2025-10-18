import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import wandb

# Import our custom modules
import config
from scripts.dataset import CAMELSDataset
from scripts.model import UNet

def train_model():
    """Main function to orchestrate the model training process."""
    wandb.init(
        project="bayronik-emulator",
        config={
            "learning_rate": config.LEARNING_RATE,
            "architecture": "U-Net",
            "dataset": f"{config.SUITE}-{config.DATASET_TYPE}",
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
        }
    )
    print(f"--- Starting Training on {config.DEVICE} ---")
    print("--- Weights & Biases tracking is enabled ---")

    # 1. Load Data
    print("Loading dataset...")
    full_dataset = CAMELSDataset(
        root_dir=config.DATA_DIR,
        suite=config.SUITE,
        dataset_type=config.DATASET_TYPE
    )
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False
    )
    print(f"Data loaded. Training size: {train_size}, Validation size: {val_size}")

    # 2. Initialize Model, Loss, and Optimizer
    model = UNet(n_channels=1, n_classes=1).to(config.DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 3. Training Loop with Validation
    print("\n--- Entering Training Loop ---")
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for data, targets in train_loader:
            data = data.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        # --- Logging ---
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(config.WEIGHTS_DIR, f"best_{config.MODEL_NAME}")
            os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved (val_loss: {avg_val_loss:.6f})")

    print("\n--- Training Finished ---")
    print("--- Proceeding to save model weights. ---")

    try:
        # --- Save the final model weights ---
        print("Step 1: Creating weights directory (if it doesn't exist)...")
        os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
        
        model_path = os.path.join(config.WEIGHTS_DIR, config.MODEL_NAME)
        print(f"Step 2: Preparing to save model to: {model_path}")
        
        # --- THE FIX: Move model to CPU before saving ---
        print("Step 3: Moving model to CPU for safe saving...")
        model.to("cpu")
        
        torch.save(model.state_dict(), model_path)
        
        print("Step 4: torch.save command executed successfully.")
        print(f"Model saved to {model_path}")

    except Exception as e:
        print("\n---! AN ERROR OCCURRED DURING SAVING !---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("The model was NOT saved.")
    
    # --- Finalize W&B run ---
    print("\nFinalizing Weights & Biases run...")
    wandb.finish()
    print("W&B run finished.")

if __name__ == "__main__":
    train_model()

