"""
ProbZero Training Loop
======================
Runs the self-play → train cycle.

Usage (from repository root):
    python scripts/run.py

Requirements:
    - Build the C++ engine first (see README.md)
    - PyTorch installed
"""

import os
import sys
import time
import subprocess

# Add scripts dir to path for train.py imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from train import (
        OthelloNet,
        TreeDataset,
        train_epoch,
        save_traced,
        DEVICE,
        LR,
    )
except ImportError:
    # Fallback if run directly from scripts/
    sys.path.append(".")
    from train import (
        OthelloNet,
        TreeDataset,
        train_epoch,
        save_traced,
        DEVICE,
        LR,
    )

# ================== CONFIG ==================
# Paths relative to repository root (assuming run from root)
REPO_ROOT = os.path.dirname(current_dir)
ENGINE_PATH = os.path.join(REPO_ROOT, "build", "c4_engine")

START_ITER = 0
SAVE_EVERY = 50
MAX_ITERATIONS = 1800

MODEL_LATEST = os.path.join(REPO_ROOT, "model_latest.pt")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset.bin")

CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 512
SLEEP_SECONDS = 1
# ============================================


def run_selfplay(model_path, dataset_path):
    """Run C++ self-play engine."""
    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(f"Engine not found at {ENGINE_PATH}. Please run 'mkdir -p build && cd build && cmake .. && make -j' first.")
        
    cmd = [ENGINE_PATH, model_path, dataset_path]
    subprocess.run(cmd, check=True)


def load_traced_weights_into_model(model, traced_path):
    """
    Load weights from a TorchScript model into a raw PyTorch model.
    Handles wrapped models (e.g. traced.m.*).
    """
    if not os.path.exists(traced_path):
        print(f"⚠ Warning: Resume model {traced_path} not found.")
        return

    traced = torch.jit.load(traced_path, map_location=DEVICE)

    if hasattr(traced, "m"):
        state = traced.m.state_dict()
    else:
        state = traced.state_dict()

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"⚠ Missing keys: {missing}")
    if unexpected:
        print(f"⚠ Unexpected keys: {unexpected}")

    print(f"✔ Model weights loaded successfully from {traced_path}")


def main():
    print(f"=== ProbZero Training on {DEVICE} ===")
    print(f"Starting from iteration {START_ITER}")

    # -------- INIT MODEL --------
    model = OthelloNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------- OPTIONAL: RESUME FROM CHECKPOINT --------
    # Uncomment to resume from a saved model:
    # RESUME_ITER = 300
    # RESUME_PATH = os.path.join(CHECKPOINT_DIR, f"model_iter_{RESUME_ITER}.pt")
    # load_traced_weights_into_model(model, RESUME_PATH)
    # START_ITER = RESUME_ITER

    # Save initial model for engine
    save_traced(model, MODEL_LATEST)

    cycle = START_ITER

    # -------- TRAIN LOOP --------
    while True:
        cycle += 1
        print(f"\n=== Cycle {cycle} ===")

        # ----- SELF-PLAY -----
        print("Running self-play...")
        if os.path.exists(DATASET_PATH):
            os.remove(DATASET_PATH)

        try:
            run_selfplay(MODEL_LATEST, DATASET_PATH)
        except subprocess.CalledProcessError:
            print("❌ Engine crashed — skipping cycle")
            continue

        if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) < 1000:
            print("⚠ No valid data generated")
            continue

        # ----- TRAIN -----
        dataset = TreeDataset(DATASET_PATH)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        train_epoch(model, optimizer, loader, cycle)

        # ----- SAVE LATEST -----
        save_traced(model, MODEL_LATEST)

        # ----- PERIODIC CHECKPOINT -----
        if cycle % SAVE_EVERY == 0:
            weights_path = os.path.join(CHECKPOINT_DIR, f"weights_iter_{cycle}.pth")
            model_path = os.path.join(CHECKPOINT_DIR, f"model_iter_{cycle}.pt")

            torch.save(model.state_dict(), weights_path)
            save_traced(model, model_path)

            print(f"💾 Saved checkpoint at iteration {cycle}")

        # ----- CLEANUP -----
        os.remove(DATASET_PATH)
        time.sleep(SLEEP_SECONDS)

        if cycle >= MAX_ITERATIONS:
            print(f"\n=== Training complete ({MAX_ITERATIONS} iterations) ===")
            break


if __name__ == "__main__":
    main()
