import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys

# --- CONFIG ---
BATCH_SIZE = 512
LR = 0.0005 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)

class OthelloNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=50):
        super(OthelloNet, self).__init__()
        
        # 1. Initial Convolutional Block
        self.conv_input = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 2. Residual Tower (The backbone)
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # 3. Policy Head (Standard AlphaZero style)
        # We use a 1x1 conv to reduce channels before the linear layer to prevent overfitting
        self.p_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 5 * 5, 26) 

        # 4. Value Head (Categorical: Loss, Draw, Win)
        self.v_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(32)
        self.v_fc1 = nn.Linear(32 * 5 * 5, num_channels)
        self.v_fc2 = nn.Linear(num_channels, 3) 

        # 5. Reward Head (Auxiliary)
        self.r_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.r_bn = nn.BatchNorm2d(32)
        self.r_fc1 = nn.Linear(32 * 5 * 5, num_channels)
        self.r_fc2 = nn.Linear(num_channels, 3)

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p) # Outputs 26 logits

        # Value Head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = self.v_fc2(v) # Outputs 3 (Categorical)

        # Reward Head
        r = F.relu(self.r_bn(self.r_conv(x)))
        r = r.view(r.size(0), -1)
        r = F.relu(self.r_fc1(r))
        r = self.r_fc2(r) # Outputs 3 (Categorical)
        
        return p, v, r
# --- 2. DATASET LOADER ---
# --- 2. DATASET LOADER ---
class TreeDataset(Dataset):
    def __init__(self, filename):
        if not os.path.exists(filename):
            print(f"Dataset {filename} not found.")
            self.len = 0
            return
            
        with open(filename, "rb") as f:
            try:
                header = f.read(4)
                if not header:
                    self.len = 0
                    return
                self.len = np.frombuffer(header, dtype=np.int32)[0]
                raw = f.read()
            except Exception as e:
                print(f"Error reading dataset: {e}")
                self.len = 0
                return
        
        # Stride = Board(50) + Policy(26) + Mask(26) + V(3) + R(3) + Valid(1) + Weight(1) = 266
        floats_per_sample = 110
        
        if len(raw) // 4 != self.len * floats_per_sample:
            print(f"Size mismatch. Recalculating...")
            self.len = len(raw) // 4 // floats_per_sample

        print(f"Loading {self.len} samples from {filename}...")
        arr = np.frombuffer(raw, dtype=np.float32).reshape(self.len, floats_per_sample)
        arr = np.nan_to_num(arr, nan=0.0)

        self.boards   = arr[:, 0:50].reshape(self.len, 2, 5, 5)
        self.policies = arr[:, 50:76]
        self.p_masks  = arr[:, 76:102]
        self.target_q = arr[:, 102:105]
        self.target_r = arr[:, 105:108]
        self.r_valid  = arr[:, 108]
        self.weights  = arr[:, 109]
        
        self.print_stats()

    def print_stats(self):
        if self.len == 0: return
        print("\n=== DATASET STATS ===")
        print(f"Total Samples: {self.len}")
        
        # --- 1. VALUE TARGETS (Q) ---
        # Mean: Average of the raw numbers (shows uncertainty/soft targets)
        avg_q = np.mean(self.target_q, axis=0)
        
        # Argmax: Which class is the winner? (shows what training will pick)
        argmax_q = np.argmax(self.target_q, axis=1)
        dist_q = [np.sum(argmax_q == i) / self.len for i in range(3)]
        
        print(f"\n[Value Head (Game Outcome)]")
        print(f"  Mean (Soft Avg)   : Loss {avg_q[0]*100:.1f}% | Draw {avg_q[1]*100:.1f}% | Win {avg_q[2]*100:.1f}%")
        print(f"  Argmax (Dominant) : Loss {dist_q[0]*100:.1f}% | Draw {dist_q[1]*100:.1f}% | Win {dist_q[2]*100:.1f}%")

        # --- 2. REWARD TARGETS (R) ---
        # Only check valid rewards (terminal states)
        valid_indices = (self.r_valid > 0.5)
        n_valid = np.sum(valid_indices)
        
        if n_valid > 0:
            valid_r = self.target_r[valid_indices]
            avg_r = np.mean(valid_r, axis=0)
            
            argmax_r = np.argmax(valid_r, axis=1)
            dist_r = [np.sum(argmax_r == i) / n_valid for i in range(3)]
            
            print(f"\n[Reward Head (Terminal Move)] - ({n_valid} samples found)")
            print(f"  Mean (Soft Avg)   : Loss {avg_r[0]*100:.1f}% | Draw {avg_r[1]*100:.1f}% | Win {avg_r[2]*100:.1f}%")
            print(f"  Argmax (Dominant) : Loss {dist_r[0]*100:.1f}% | Draw {dist_r[1]*100:.1f}% | Win {dist_r[2]*100:.1f}%")
        else:
            print(f"\n[Reward Head] No terminal states found in this batch (Valid=0).")
            
        print("=====================\n")

    def __len__(self): return self.len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.boards[idx]),
            torch.tensor(self.policies[idx]),
            torch.tensor(self.p_masks[idx]),
            torch.tensor(self.target_q[idx]),
            torch.tensor(self.target_r[idx]),
            torch.tensor(self.r_valid[idx]),
            torch.tensor(self.weights[idx])
        )
    
# --- 3. TRAINING WITH METRICS ---
def train_epoch(model, opt, loader, epoch):
    model.train()
    
    # Trackers
    stats = {
        "loss_total": 0.0, "loss_p": 0.0, "loss_v": 0.0, "loss_r": 0.0,
        "acc_p_top1": 0.0, "acc_v": 0.0, "acc_r": 0.0,
        "count": 0, "count_r": 0
    }
    
    # Distribution Trackers (Loss, Draw, Win)
    # [Target_L, Target_D, Target_W] vs [Pred_L, Pred_D, Pred_W]
    dist_v_target = torch.zeros(3).to(DEVICE)
    dist_v_pred   = torch.zeros(3).to(DEVICE)

    for b, tp, pm, tq, tr, rv, w in loader:
        b, tp, pm, tq, tr, rv, w = b.to(DEVICE), tp.to(DEVICE), pm.to(DEVICE), tq.to(DEVICE), tr.to(DEVICE), rv.to(DEVICE), w.to(DEVICE)
        
        # Normalize weights for stability
        w_norm = w / (w.mean() + 1e-9)

        # Forward
        pp, pv, pr = model(b)
        
        # --- LOSS CALCULATION ---
        # 1. Policy Loss (Masked)
        pp_masked = pp.clone()
        pp_masked[pm == 0] = -1e9
        loss_p_raw = -(tp * F.log_softmax(pp_masked, dim=1)).sum(dim=1)
        loss_p = (loss_p_raw * w_norm).mean()
        
        # 2. Value Loss
        loss_v_raw = -(tq * F.log_softmax(pv, dim=1)).sum(dim=1)
        loss_v = (loss_v_raw * w_norm).mean()
        
        # 3. Reward Loss (FIXED)
        ce_r = -(tr * F.log_softmax(pr, dim=1)).sum(dim=1)

        # Apply validity mask to weights first
        valid_weights = w * rv 

        # Avoid division by zero
        sum_valid_weights = valid_weights.sum() + 1e-6

        # Calculate Weighted Average properly
        loss_r = (ce_r * valid_weights).sum() / sum_valid_weights

        loss = loss_p + loss_v + loss_r
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Batch loss is {loss.item()} (Ignored)")
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # --- METRICS CALCULATION (No Grad) ---
        with torch.no_grad():
            batch_n = b.shape[0]
            stats["count"] += batch_n
            stats["loss_total"] += loss.item() * batch_n
            stats["loss_p"] += loss_p.item() * batch_n
            stats["loss_v"] += loss_v.item() * batch_n
            stats["loss_r"] += loss_r.item() * batch_n

            # A. Policy Accuracy (Did we match MCTS top move?)
            pred_move = pp_masked.argmax(dim=1)
            target_move = tp.argmax(dim=1)
            stats["acc_p_top1"] += (pred_move == target_move).sum().item()

            # B. Value Accuracy (Did we predict the outcome correctly?)
            # 0=Loss, 1=Draw, 2=Win
            pred_outcome = pv.argmax(dim=1)
            target_outcome = tq.argmax(dim=1)
            stats["acc_v"] += (pred_outcome == target_outcome).sum().item()
            
            # Accumulate distributions for visual check
            for i in range(3):
                dist_v_target[i] += (target_outcome == i).sum()
                dist_v_pred[i]   += (pred_outcome == i).sum()

            # C. Reward Accuracy (Only for valid rewards)
            valid_mask = (rv > 0.5)
            n_valid = valid_mask.sum().item()
            if n_valid > 0:
                pred_r = pr.argmax(dim=1)
                target_r = tr.argmax(dim=1)
                stats["acc_r"] += (pred_r[valid_mask] == target_r[valid_mask]).sum().item()
                stats["count_r"] += n_valid

    # --- SUMMARY PRINT ---
    def avg(k, div): return stats[k] / max(stats[div], 1)
    
    print(f"\n--- Epoch {epoch} Report ---")
    print(f"LOSS    : Total {avg('loss_total', 'count'):.4f} | Pol {avg('loss_p', 'count'):.4f} | Val {avg('loss_v', 'count'):.4f} | Rew {avg('loss_r', 'count'):.4f}")
    print(f"ACCURACY: Policy {avg('acc_p_top1', 'count')*100:.1f}%   | Value {avg('acc_v', 'count')*100:.1f}%  | Reward {avg('acc_r', 'count_r')*100:.1f}%")
    
    # Distribution Visual
    total = stats["count"]
    dt = (dist_v_target / total * 100).cpu().numpy()
    dp = (dist_v_pred / total * 100).cpu().numpy()
    
    print(f"DISTRIB : [Loss / Draw / Win]")
    print(f"  Target: [{dt[0]:4.1f}% {dt[1]:4.1f}% {dt[2]:4.1f}%]")
    print(f"  Pred  : [{dp[0]:4.1f}% {dp[1]:4.1f}% {dp[2]:4.1f}%]")
    
    # Progress check
    if dp[0] > 99 or dp[2] > 99:
        print("  WARNING: Model collapsing to single prediction!")
        
    return avg('loss_total', 'count')

# --- 4. EXPORT ---
def save_traced(model, filename="model_script.pt"):
    # CPU Export for Safety
    model.cpu().eval()
    example = torch.rand(1, 2, 5, 5)
    
    class ExportWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            p, v, r = self.m(x)
            return p, F.softmax(v, dim=1), F.softmax(r, dim=1)
            
    traced = torch.jit.trace(ExportWrapper(model), example)
    traced.save(filename)
    model.to(DEVICE) # Move back
    print(f"Saved traced model to {filename} (CPU compatible)")

if __name__ == "__main__":
    if not os.path.exists("model_script.pt"):
        print("Initializing random Othello model...")
        model = OthelloNet().to(DEVICE)
        save_traced(model)

    ds = TreeDataset("dataset.bin")
    
    if ds.len > 0:
        model = OthelloNet().to(DEVICE)
        try:
            model.load_state_dict(torch.load("weights.pth"))
            print("Loaded previous weights.")
        except: pass
        
        opt = optim.Adam(model.parameters(), lr=LR)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train for a few epochs to see stability
        for epoch in range(1, 4): 
            loss = train_epoch(model, opt, ld, epoch)
        
        save_traced(model)
        torch.save(model.state_dict(), "weights.pth")
    else:
        print("No data found. Please run C++ engine.")