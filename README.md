# ProbZero

**Reward-Bootstrapped AlphaZero with Full-Tree Policy Training**

A novel AlphaZero variant that introduces two key modifications to the standard algorithm, demonstrated on 5×5 Othello (Mini-Othello).

## Key Innovations

### 1. Reward-Bootstrapped Value Head
Instead of training the value head directly from game outcomes (as in standard AlphaZero), ProbZero:
- Trains a **reward head** on terminal states (ground-truth win/draw/loss)
- Bootstraps the **value head** using the reward head's predictions after MCTS search
- This allows **all explored positions** (not just game-played positions) to contribute meaningful value targets

### 2. Full-Tree Policy Training
Standard AlphaZero trains the policy on the root node's visit distribution. ProbZero instead:
- Extracts policy targets from **all explored nodes** in the search tree
- Only considers **explored children** for each node (unexplored moves are masked)
- Computes policy targets as WDL (Win/Draw/Loss) probabilities derived from children's values
- Weights each position's contribution by **visit count**

### 3. Particle-Based MCTS
Uses a particle-based tree expansion where "volume" (particles) flows through the tree:
- Particles are distributed among children proportionally to the current policy
- Unexplored moves collect into a "bucket" that gets expanded in priority order
- Enables efficient batched GPU inference during tree expansion

## Architecture

```
Input: 2×5×5 (self pieces, opponent pieces)
  ↓
Conv2d(2 → 50, 3×3) + BN + ReLU
  ↓
10× ResBlock(50 channels, 3×3)
  ↓
┌─────────────────┬──────────────────┬──────────────────┐
│   Policy Head   │   Value Head     │   Reward Head    │
│ Conv(50→32,1×1) │ Conv(50→32,1×1)  │ Conv(50→32,1×1)  │
│ FC(800→26)      │ FC(800→50→3)     │ FC(800→50→3)     │
│ → 26 logits     │ → 3 (L,D,W)     │ → 3 (L,D,W)     │
└─────────────────┴──────────────────┴──────────────────┘
```

## Project Structure

```
ProbZero/
├── src/
│   ├── game.hpp          # 5×5 Othello engine (bitboard)
│   ├── node.hpp          # MCTS tree node
│   ├── mcts.cpp/.hpp     # MCTS with particle-based search
│   ├── model.hpp         # LibTorch model wrapper
│   ├── selfplay.hpp      # Self-play data generation
│   ├── main.cpp          # Self-play binary (data generator)
│   └── play.cpp          # Arena binary (model vs model)
├── train.py              # PyTorch training loop
├── auto_loop.py          # Automated self-play → train loop
├── master.py             # Distributed training coordinator
├── worker.py             # Distributed self-play worker
├── arena.py              # Elo tournament runner
├── convert_pt.py         # Weight → TorchScript converter
├── elo_estimation.py     # Kaggle-ready Elo estimation script
├── CMakeLists.txt        # C++ build configuration
├── CITATION.cff          # Citation metadata
└── LICENSE               # Apache 2.0
```

## Training Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Self-Play   │────▶│  Train.py    │────▶│  Export .pt   │
│  (C++ MCTS)  │     │  (PyTorch)   │     │  (TorchScript)│
│  dataset.bin │     │  weights.pth │     │  model.pt     │
└──────────────┘     └──────────────┘     └───────┬───────┘
       ▲                                          │
       └──────────────────────────────────────────┘
                    (loop)
```

### Data Format (Binary)
Each sample: `Board(50) + Policy(26) + Mask(26) + Value(3) + Reward(3) + Valid(1) + Weight(1)` = 110 floats

### Loss Function
```
L = L_policy + L_value + L_reward

L_policy = -Σ(target_p · log_softmax(pred_p))    [masked to explored children]
L_value  = -Σ(target_q · log_softmax(pred_v))    [categorical cross-entropy]
L_reward = -Σ(target_r · log_softmax(pred_r))    [only terminal states]
```
All losses are weighted by per-sample visit count.

## Building

### Requirements
- LibTorch (C++17)
- PyTorch (Python)
- CUDA (recommended) or CPU

### Compile
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j$(nproc)
```

### Quick Start
```bash
# 1. Initialize a random model
python train.py

# 2. Run self-play (generates dataset.bin)
./build/c4_engine model_script.pt dataset.bin 60 20

# 3. Train on generated data
python train.py

# 4. Repeat (or use auto_loop.py)
python auto_loop.py
```

### Arena (Model vs Model)
```bash
# AI vs AI battle (20 games, temp=1.0)
./build/play 4 model_a.pt model_b.pt 20 1.0
```

## Results

Training on 5×5 Othello shows clear Elo progression across iterations. See `elo_estimation.py` for reproduction.

## Citation

If you use this work, please cite:

```bibtex
@software{probzero2025,
  title={ProbZero: Reward-Bootstrapped AlphaZero with Full-Tree Policy Training},
  author={REPLACE_WITH_YOUR_NAME},
  year={2025},
  url={https://github.com/REPLACE_WITH_YOUR_USERNAME/ProbZero},
  license={Apache-2.0}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
