# ProbZero

**Reward-Bootstrapped AlphaZero with Particle-Based MCTS**

A novel AlphaZero variant that introduces reward bootstrapping for value training and full-tree policy extraction, demonstrated on 5×5 Othello (Mini-Othello).

---

## Key Contributions

### 1. Reward-Bootstrapped Value Training
Standard AlphaZero trains the value head from game outcomes (terminal reward only). ProbZero separates this into two concepts:

- **Reward Head**: Trained on terminal game outcomes (ground-truth WDL). The "reward" here refers to the **actual outcome observed at terminal nodes in the MCTS tree**, not the reinforcement learning notion of per-step reward signals.
- **Value Head**: Bootstrapped from the reward head's predictions, propagated backwards through the search tree via `winner_prob` weighting. This means positions never actually reached at game end can still receive meaningful value targets from nearby terminal evaluations.

> **Note**: Reward bootstrapping as a general technique is well-established in RL. The contribution here is applying it within the MCTS tree structure, where a dedicated **reward head** predicts terminal outcomes and these predictions are propagated to non-terminal nodes as value targets through the search tree's structure.

### 2. Full-Tree Policy Training
Rather than training policy targets from only the root node's visit distribution (as in standard AlphaZero):

- Policy targets are extracted from **every explored node** in the search tree
- Only **explored children** are considered (unexplored moves are masked out)
- Targets are computed as WDL probabilities via `winner_prob` over children's values
- Each position's contribution is weighted by its **visit count**

### 3. Particle-Based MCTS
Uses a particle-based tree expansion instead of single-path UCB traversal:

- "Particles" (volume) flow through the tree each simulation batch
- Particles distribute among children proportionally to current policy
- Unexplored moves collect into a "bucket" expanded in policy-priority order
- Enables **batched GPU inference** — all leaf nodes from one batch are evaluated simultaneously

---

## Architecture

### Neural Network

```
Input: 2×5×5 (current player pieces, opponent pieces)
  │
  ▼
Conv2d(2 → 50, 3×3, pad=1) + BatchNorm + ReLU
  │
  ▼
10× Residual Blocks
  │    ┌─────────────────────────────┐
  │    │ Conv(50→50, 3×3) + BN + ReLU│
  │    │ Conv(50→50, 3×3) + BN       │
  │    │ + Skip Connection + ReLU     │
  │    └─────────────────────────────┘
  │
  ├──────────────────┬──────────────────┐
  ▼                  ▼                  ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│Policy Head │ │ Value Head │ │Reward Head │
│Conv(1×1)   │ │Conv(1×1)   │ │Conv(1×1)   │
│BN + ReLU   │ │BN + ReLU   │ │BN + ReLU   │
│Flatten     │ │Flatten     │ │Flatten     │
│FC(800→26)  │ │FC(800→50)  │ │FC(800→50)  │
│            │ │ReLU        │ │ReLU        │
│→ 26 logits │ │FC(50→3)    │ │FC(50→3)    │
│(25 squares │ │→ WDL probs │ │→ WDL probs │
│ + 1 pass)  │ │(softmax)   │ │(softmax)   │
└────────────┘ └────────────┘ └────────────┘
```

### Particle-Based MCTS Search

```mermaid
flowchart TD
    A["Root Node<br/>volume = 16 particles"] --> B{"Distribute particles<br/>proportional to policy"}
    B --> C["Child A<br/>(explored, p=0.5)<br/>→ 8 particles"]
    B --> D["Child B<br/>(explored, p=0.2)<br/>→ 3 particles"]
    B --> E["Bucket<br/>(unexplored, p=0.3)<br/>→ 5 particles"]
    
    C --> F["Recurse deeper<br/>(8 particles)"]
    D --> G["Recurse deeper<br/>(3 particles)"]
    E --> H{"Expand in<br/>policy-priority order"}
    H --> I["New Child C<br/>1 particle → leaf"]
    H --> J["New Child D<br/>1 particle → leaf"]
    H --> K["New Child E<br/>1 particle → leaf"]
    H --> L["... remaining 2<br/>particles allocated"]
    
    I --> M["Batch NN inference<br/>on all leaves"]
    J --> M
    K --> M
    F --> M
    G --> M
    
    M --> N["Backpropagate:<br/>forward_update"]

    style A fill:#2d5a27,color:#fff
    style M fill:#1a4a6e,color:#fff
    style N fill:#6e1a1a,color:#fff
```

### Winner Probability Calculation (`winner_prob`)

The `winner_prob` function converts children's WDL (Win/Draw/Loss) value estimates into a probability distribution over which child is the "winning" move. This is used to update `policy_search` during backpropagation.

For each child *i* with value estimates (W_i, D_i, L_i) from the **parent's perspective** (so child's Win = parent's Loss):

```
For child i:
  p0_i = child's Win probability  (= parent's Loss if chosen)
  p1_i = child's Draw probability
  p2_i = child's Loss probability (= parent's Win if chosen)

Compute products (excluding self):
  prod_lt1 = ∏ (p0_j + ε)  for all j
  prod_lt2 = ∏ (p0_j + p1_j + ε)  for all j

For each child i:
  term1_i = p1_i × prod_lt1 / (p0_i + ε)     ← "I draw, all others lose"
  term2_i = p2_i × prod_lt2 / (p0_i + p1_i + ε)  ← "I win outright"
  
  raw_win_i = term1_i + term2_i

Normalize: weight_i = raw_win_i / Σ raw_win_j
```

**Intuition**: A child gets high weight if (a) the opponent loses when playing it (high p2, meaning parent wins), or (b) the opponent draws while all other children would lose. This selects moves that maximize the parent's winning chances.

### Target Calculation Pipeline

```mermaid
flowchart TB
    subgraph TerminalNodes["Terminal Nodes (game over)"]
        T1["Terminal: Black wins<br/>reward = (1, 0, 0)"]
        T2["Terminal: Draw<br/>reward = (0, 1, 0)"]
        T3["Terminal: White wins<br/>reward = (0, 0, 1)"]
    end
    
    subgraph RewardProp["Reward Propagation (final_update)"]
        direction TB
        R1["For each parent node:"]
        R2["1. Collect explored children"]
        R3["2. Compute winner_prob weights"]
        R4["3. reward_target = Σ w_i × child.reward_target<br/>(with W↔L flip for opponent perspective)"]
        R1 --> R2 --> R3 --> R4
    end
    
    subgraph ValueProp["Value Propagation (forward_update)"]
        direction TB
        V1["For each node during search:"]
        V2["1. Compute winner_prob weights"]
        V3["2. policy_search = weights × explored_mass"]
        V4["3. value_search = Σ w_i × child.value_search<br/>(with W↔L flip)"]
        V5["4. value_target = Σ w_i × child.value_target<br/>(with W↔L flip)"]
        V1 --> V2 --> V3 --> V4 --> V5
    end
    
    subgraph Training["Training Targets (per node)"]
        direction TB
        L1["Policy target: policy_search<br/>(only explored children, masked)"]
        L2["Value target: value_target<br/>(bootstrapped from reward head)"]
        L3["Reward target: reward_target<br/>(propagated from terminals)<br/>valid only if terminal path exists"]
        L4["Weight: visit_count"]
    end
    
    TerminalNodes --> RewardProp
    RewardProp --> Training
    ValueProp --> Training

    style TerminalNodes fill:#1a4a6e,color:#fff
    style RewardProp fill:#6e1a1a,color:#fff
    style ValueProp fill:#2d5a27,color:#fff
    style Training fill:#4a3a6e,color:#fff
```

### Loss Function

```
L = L_policy + L_value + L_reward

L_policy = -Σ (target_p · log_softmax(pred_p))    [masked to explored children only]
L_value  = -Σ (target_q · log_softmax(pred_v))    [categorical cross-entropy, WDL]
L_reward = -Σ (target_r · log_softmax(pred_r))    [only for nodes with terminal paths]

All losses weighted by per-sample visit count.
```

---

## Results

Training on 5×5 Othello shows clear Elo progression across self-play iterations:

![Elo progression across training iterations (5×5 Othello)](elo_progression.png)

Later iterations consistently dominate earlier ones, with iter_400 achieving ~359 Elo above the baseline (iter_50). See `elo_estimation.py` for full reproduction.

---

## Project Structure

```
ProbZero/
├── src/
│   ├── game.hpp          # 5×5 Othello engine (bitboard)
│   ├── node.hpp          # MCTS tree node definition
│   ├── mcts.cpp/.hpp     # Particle-based MCTS with winner_prob
│   ├── model.hpp         # LibTorch model wrapper
│   ├── selfplay.hpp      # Self-play data generation (full tree extraction)
│   ├── main.cpp          # Self-play binary (data generator)
│   └── play.cpp          # Arena binary (model vs model)
├── train.py              # PyTorch training loop
├── auto_loop.py          # Automated self-play → train loop
├── master.py             # Distributed training coordinator
├── worker.py             # Distributed self-play worker
├── arena.py              # Elo tournament runner
├── convert_pt.py         # Weight → TorchScript converter
├── elo_estimation.py     # Kaggle-ready Elo estimation script
├── 5x5_othello_models/   # Pre-trained model checkpoints
├── CMakeLists.txt        # C++ build configuration
├── CITATION.cff          # Citation metadata
└── LICENSE               # Apache 2.0
```

---

## Training Pipeline

```mermaid
flowchart LR
    A["Self-Play<br/>(C++ MCTS)"] -->|"dataset.bin<br/>(all tree positions)"| B["Train<br/>(PyTorch)"]
    B -->|"weights.pth"| C["Export<br/>TorchScript"]
    C -->|"model.pt"| A
    
    style A fill:#2d5a27,color:#fff
    style B fill:#1a4a6e,color:#fff
    style C fill:#4a3a6e,color:#fff
```

### Data Format (Binary)
Each sample: `Board(50) + Policy(26) + Mask(26) + Value(3) + Reward(3) + Valid(1) + Weight(1)` = 110 floats

---

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
# AI vs AI parallel arena (20 games)
./build/play 4 model_a.pt model_b.pt 20
```

---

## Future Work

- **Replay Buffer**: Implement experience replay to reuse training data across iterations, improving sample efficiency
- **Extended Training**: Scale to more self-play iterations with longer training schedules
- **8×8 Othello**: Adapt the architecture and search to standard Othello, requiring deeper residual networks and more MCTS simulations — will need significant compute scaling
- **Hyperparameter Tuning**: Systematic exploration of particle volume, batch size, temperature scheduling, and `winner_prob` temperature

---

## Citation

If you use this work, please cite:

```bibtex
@software{probzero2026,
  title={ProbZero: Reward-Bootstrapped AlphaZero with Particle-Based MCTS},
  author={Mothish M},
  year={2026},
  url={https://github.com/chrono-motz/ProbZero},
  license={Apache-2.0}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
