"""
ProbZero 4x4 Training Loop
===========================
Runs the self-play → train cycle for 4x4 Othello.

Usage (from ProbZero4x4 root):
    python scripts/run.py

Requirements:
    - Build the C++ engine first:
        mkdir -p build && cd build && cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch && make -j
    - PyTorch installed
"""

import os
import sys
import time
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from train import OthelloNet, TreeDataset, train_epoch, save_traced, DEVICE, LR
except ImportError:
    sys.path.append(".")
    from train import OthelloNet, TreeDataset, train_epoch, save_traced, DEVICE, LR

# ================== CONFIG ==================
REPO_ROOT   = os.path.dirname(current_dir)
ENGINE_PATH = os.path.join(REPO_ROOT, "build", "c4_engine")

START_ITER      = 0
SAVE_EVERY      = 10
MAX_ITERATIONS  = 1800

MODEL_LATEST  = os.path.join(REPO_ROOT, "model_latest.pt")
DATASET_PATH  = os.path.join(REPO_ROOT, "dataset.bin")

CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE    = 512
SLEEP_SECONDS = 1
# ============================================


def run_selfplay(model_path, dataset_path, cycle):
    if not os.path.exists(ENGINE_PATH):
        raise FileNotFoundError(
            f"Engine not found at {ENGINE_PATH}. "
            "Please build first: mkdir -p build && cd build && cmake .. && make -j"
        )
    cmd = [ENGINE_PATH, model_path, dataset_path]
    env = os.environ.copy()
    env["ITERATION"] = str(cycle)
    subprocess.run(cmd, env=env, check=True)


def load_traced_weights_into_model(model, traced_path):
    if not os.path.exists(traced_path):
        print(f"⚠ Warning: Resume model {traced_path} not found.")
        return
    traced = torch.jit.load(traced_path, map_location=DEVICE)
    state  = traced.m.state_dict() if hasattr(traced, "m") else traced.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:     print(f"⚠ Missing keys: {missing}")
    if unexpected:  print(f"⚠ Unexpected keys: {unexpected}")
    print(f"✔ Weights loaded from {traced_path}")


def main():
    print(f"=== ProbZero 4x4 Training on {DEVICE} ===")
    print(f"Starting from iteration {START_ITER}")

    model     = OthelloNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Uncomment to resume:
    # RESUME_ITER = 100
    # load_traced_weights_into_model(model, os.path.join(CHECKPOINT_DIR, f"model_iter_{RESUME_ITER}.pt"))

    save_traced(model, MODEL_LATEST)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "weights_iter_0.pth"))
    save_traced(model,             os.path.join(CHECKPOINT_DIR, "model_iter_0.pt"))

    cycle = START_ITER

    while True:
        cycle += 1
        print(f"\n=== Cycle {cycle} ===")

        print("Running self-play...")
        if os.path.exists(DATASET_PATH):
            os.remove(DATASET_PATH)

        try:
            run_selfplay(MODEL_LATEST, DATASET_PATH, cycle)
        except subprocess.CalledProcessError:
            print("❌ Engine crashed — skipping cycle")
            continue

        if not os.path.exists(DATASET_PATH) or os.path.getsize(DATASET_PATH) < 1000:
            print("⚠ No valid data generated")
            continue

        dataset = TreeDataset(DATASET_PATH)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_epoch(model, optimizer, loader, cycle)

        save_traced(model, MODEL_LATEST)

        if cycle % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"weights_iter_{cycle}.pth"))
            save_traced(model,             os.path.join(CHECKPOINT_DIR, f"model_iter_{cycle}.pt"))
            print(f"💾 Saved checkpoint at iteration {cycle}")

        os.remove(DATASET_PATH)
        time.sleep(SLEEP_SECONDS)

        if cycle >= MAX_ITERATIONS:
            print(f"\n=== Training complete ({MAX_ITERATIONS} iterations) ===")
            break


if __name__ == "__main__":
    main()
