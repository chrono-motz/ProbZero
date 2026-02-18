import os
import time
import shutil
import subprocess
import torch
from train import OthelloNet, save_traced, DEVICE

# --- CONFIG ---
WORKER_ID = "worker_1" # Auto-assigned if possible, or manual
DRIVE_PATH = "/content/drive/MyDrive/OthelloZero"
LOCAL_MODEL = "model_script.pt"
LOCAL_DATA = "dataset.bin"
ENGINE_PATH = "./build/c4_engine" # Originally was c4_engine, but user prompt says c4_engine. 
# NOTE: User prompt said "ENGINE_PATH = ./build/c4_engine" but used "auto_loop.py" which usually builds "play"
# Whatever, I will use c4_engine as per user script, assuming main.cpp builds into it.

# Check if we need to detect ID from environment?
if "COLAB_GPU" in os.environ:
   # Just a simple hack to randomize ID slightly or assume user sets it
   import random
   WORKER_ID = f"worker_{random.randint(100,999)}"

def main():
    print(f"=== {WORKER_ID} STARTED ===")

    # Check if engine exists
    if not os.path.exists(ENGINE_PATH) and os.path.exists("./build/play"):
         # Fallback if user compiled 'play' but script asks 'c4_engine'
         # But wait, main.cpp builds c4_engine. play.cpp builds play.
         # Selfplay is usually main.cpp (c4_engine).
         pass

    if not os.path.exists(ENGINE_PATH):
        print(f"Error: Engine not found at {ENGINE_PATH}")
        print("Run: mkdir -p build && cd build && cmake .. && make")
        return

    while True:
        # 1. GET LATEST MODEL
        # We copy from Drive to local to avoid latency during play
        drive_model = f"{DRIVE_PATH}/best_model.pt"
        
        # Retry loop for model
        tries = 0
        while not os.path.exists(drive_model):
            print("Waiting for model in Drive...")
            time.sleep(10)
            tries += 1
            if tries > 5:
                # If no model exists, maybe we are the first worker?
                # Cannot create it easily. Master should do it.
                pass

        if os.path.exists(drive_model):
            # Only copy if it's newer or we don't have one
            try:
                if not os.path.exists(LOCAL_MODEL) or \
                   os.path.getmtime(drive_model) > os.path.getmtime(LOCAL_MODEL):
                    print("Found new model! Downloading...")
                    shutil.copy(drive_model, LOCAL_MODEL)
            except Exception as e:
                print(f"Download invalid: {e}")

        # 2. RUN SELF-PLAY (Generate Data)
        print("Running Self-Play...")
        # main.cpp args: [model_path] [output_file] [batches] [games]
        cmd = [ENGINE_PATH, LOCAL_MODEL, LOCAL_DATA, "20", "10"] 
        # Using 20 batches (small) to report frequently? 
        # Or stick to user defaults. User had [ENGINE_PATH, LOCAL_MODEL, LOCAL_DATA]
        # main.cpp defaults: 40 batches, 10 games.
        
        try:
            # capture_output=True silences the console (stores log in memory instead)
            # check=True raises an error if the engine crashes
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        except subprocess.CalledProcessError as e:
            # ONLY print if it crashed
            print(f"CRITICAL: Engine Crashed!")
            print(e.stderr)
            print(e.stdout)
            time.sleep(5)
            continue

        # 3. UPLOAD DATA
        if os.path.exists(LOCAL_DATA) and os.path.getsize(LOCAL_DATA) > 1000:
            timestamp = int(time.time())
            # Save as: incoming_data/data_worker1_170999.bin
            dest = f"{DRIVE_PATH}/incoming_data/data_{WORKER_ID}_{timestamp}.bin"
            try:
                if not os.path.exists(f"{DRIVE_PATH}/incoming_data"):
                    os.makedirs(f"{DRIVE_PATH}/incoming_data", exist_ok=True)
                shutil.move(LOCAL_DATA, dest)
                print(f"Uploaded {dest}")
            except Exception as e:
                print(f"Upload failed: {e}")

        # 4. Cleanup & Repeat
        if os.path.exists(LOCAL_DATA): os.remove(LOCAL_DATA)
        print("Sleeping 5s...")
        time.sleep(5)

if __name__ == "__main__":
    main()