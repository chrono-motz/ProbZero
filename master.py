import os
import time
import glob
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from train import OthelloNet, TreeDataset, train_epoch, save_traced, DEVICE, BATCH_SIZE, LR

# --- CONFIG ---
DRIVE_PATH = "/content/drive/MyDrive/OthelloZero"
INCOMING_DIR = f"{DRIVE_PATH}/incoming_data"
MODEL_PATH = "model_script.pt"
MERGED_DATA = "merged_dataset.bin"
MIN_FILES_TO_TRAIN = 2 
MAX_FILES_TO_TRAIN = 5 # Increased slightly

def check_model_health(model):
    """Returns True if model is healthy (no NaNs), False otherwise."""
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False
    return True

def merge_datasets(file_list, output_path):
    print(f"Merging {len(file_list)} datasets...")
    total_samples = 0
    buffer_body = bytearray()

    for fpath in file_list:
        try:
            with open(fpath, "rb") as f:
                header = f.read(4)
                if not header: continue
                count = np.frombuffer(header, dtype=np.int32)[0]
                if count <= 0: continue # Skip empty
                total_samples += count
                buffer_body.extend(f.read())
        except Exception as e:
            print(f"Skipping corrupt file {fpath}: {e}")

    # Write Merged File
    with open(output_path, "wb") as f:
        f.write(np.array([total_samples], dtype=np.int32).tobytes())
        f.write(buffer_body)

    print(f"Merged Total: {total_samples} samples.")
    return total_samples

def main():
    print(f"=== MASTER TRAINER STARTED ON {DEVICE} ===")

    model = OthelloNet().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    weights_path = f"{DRIVE_PATH}/master_weights.pth"

    # 1. LOAD OR INIT MODEL
    loaded = False
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=DEVICE)
            model.load_state_dict(state)
            if check_model_health(model):
                print("Loaded saved weights successfully.")
                loaded = True
            else:
                print("WARNING: Saved model has NaNs! Discarding and starting fresh.")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting fresh.")
    
    if not loaded:
        print("Initializing new random model...")
        torch.save(model.state_dict(), weights_path)
        save_traced(model, MODEL_PATH)
        if not os.path.exists(DRIVE_PATH): os.makedirs(DRIVE_PATH, exist_ok=True)
        shutil.copy(MODEL_PATH, f"{DRIVE_PATH}/best_model.pt")

    # Ensure directories exist
    if not os.path.exists(INCOMING_DIR): os.makedirs(INCOMING_DIR, exist_ok=True)

    cycle = 0
    while True:
        # A. WAIT FOR DATA
        data_files = glob.glob(f"{INCOMING_DIR}/*.bin")
        if len(data_files) < MIN_FILES_TO_TRAIN:
            print(f"Waiting for data... ({len(data_files)}/{MIN_FILES_TO_TRAIN})")
            time.sleep(10)
            continue

        cycle += 1
        print(f"\n[ Cycle {cycle} ] Processing {len(data_files)} files...")

        # B. MERGE DATA
        try:
            n_samples = merge_datasets(data_files[:min(len(data_files),MAX_FILES_TO_TRAIN)], MERGED_DATA)
        except Exception as e:
            print(f"Merge Critical Error: {e}")
            time.sleep(5)
            continue

        # C. TRAIN
        if n_samples > 0:
            ds = TreeDataset(MERGED_DATA)
            if len(ds) == 0:
                print("Dataset invalid (0 samples). Cleaning up files.")
                # Fall through to cleanup
            else:
                loader = DataLoader(ds, batch_size=2048, shuffle=True)

                # Train
                train_epoch(model, opt, loader, cycle)

                # Validate Health
                if not check_model_health(model):
                    print("CRITICAL: Model collapsed (NaNs) during training! Reloading last good weights...")
                    if os.path.exists(weights_path):
                        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                        # Skip saving this bad state
                else:
                    # D. UPDATE MODEL
                    torch.save(model.state_dict(), weights_path)
                    save_traced(model, MODEL_PATH)
                    shutil.copy(MODEL_PATH, f"{DRIVE_PATH}/best_model.pt")
                    print(">>> Broadcasted new model to Workers.")

            # E. CLEANUP PROCESSED FILES
            print(f"Deleting processed data files...")
            for f in data_files[:min(len(data_files),MAX_FILES_TO_TRAIN)]:
                try:
                    if os.path.exists(f): os.remove(f)
                except Exception as e:
                    print(f"Delete failed {f}: {e}")

        else:
            print("Merged dataset was empty. Deleting empty inputs...")
            for f in data_files:
                try: os.remove(f) 
                except: pass

if __name__ == "__main__":
    main()