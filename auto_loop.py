import os
import subprocess
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components from your train.py
from train import OthelloNet, TreeDataset, train_epoch, save_traced, DEVICE, BATCH_SIZE, LR

# --- CONFIGURATION ---
TOTAL_CYCLES = 100        # How many times to repeat the loop
EPOCHS_PER_CYCLE = 1      # Train 1 epoch per new dataset (as requested)
SAVE_INTERVAL = 99        # Save a backup model every X cycles
ENGINE_PATH = "./build/c4_engine"
MODEL_PATH = "model_script.pt"
DATA_PATH = "dataset.bin"

def main():
    print(f"=== Starting Auto-Training Loop on {DEVICE} ===")
    
    # 1. Initialize Model
    model = OthelloNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Resume from previous weights if they exist
    if os.path.exists("weights.pth"):
        try:
            model.load_state_dict(torch.load("weights.pth", map_location=DEVICE))
            print("Loaded existing weights from 'weights.pth'. Continuing training...")
        except Exception as e:
            print(f"Warning: Could not load weights.pth ({e}). Starting fresh.")

    # Ensure we have a traced model for the C++ engine to start with
    if not os.path.exists(MODEL_PATH):
        print("Generatng initial C++ model...")
        save_traced(model, MODEL_PATH)

    # --- MAIN LOOP ---
    for cycle in range(1, TOTAL_CYCLES + 1):
        print(f"\n[ Cycle {cycle}/{TOTAL_CYCLES} ]")

        # A. Cleanup Old Data
        if os.path.exists(DATA_PATH):
            os.remove(DATA_PATH)

        # B. Run Self-Play (C++)
        # Arguments: ./c4_engine <model> <output> <optional: num_batches>
        # Note: You can add "10" at the end to force 10 batches if you didn't update main.cpp defaults
        cmd = [ENGINE_PATH, MODEL_PATH, DATA_PATH] 
        
        try:
            # capture_output=True hides the C++ spam (equivalent to > /dev/null)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("CRITICAL ERROR: C++ Engine Crashed!")
            print(e.stderr) # Print the error so we know why
            # break
            continue
        except FileNotFoundError:
            print(f"Error: Could not find engine at {ENGINE_PATH}. Did you compile it?")
            break

        # C. Train (Python)
        # TreeDataset will automatically handle the loading and stats printing
        if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) < 100:
            print("Warning: Dataset not generated (empty file). Skipping training.")
            continue

        ds = TreeDataset(DATA_PATH)
        if ds.len > 0:
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
            
            # Train for 1 epoch as requested
            for _ in range(EPOCHS_PER_CYCLE):
                train_epoch(model, optimizer, loader, cycle)
            
            # D. Save Progress
            # 1. Save Weights (for Python resuming)
            torch.save(model.state_dict(), "weights.pth")
            
            # 2. Save Traced Model (for C++ engine next cycle)
            save_traced(model, MODEL_PATH)
            
            # 3. Periodic Backup
            if cycle % SAVE_INTERVAL == 0:
                backup_name = f"weights_cycle_{cycle}.pth"
                torch.save(model.state_dict(), backup_name)
                print(f"Checkpoint saved: {backup_name}")

if __name__ == "__main__":
    main()