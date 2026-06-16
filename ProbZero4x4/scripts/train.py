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
        out += residual
        return F.relu(out)


class OthelloNet(nn.Module):
    """
    Optimized Neural network for 4x4 Othello.
    Input:  (batch, 2, 4, 4)
    Policy: 17 logits  (16 squares + 1 pass)
    Value:  3  logits  (Loss, Draw, Win)
    Reward: 3  logits  (Loss, Draw, Win)
    """
    def __init__(self, num_res_blocks=3, num_channels=32):
        super(OthelloNet, self).__init__()

        # 1. Initial Convolutional Block
        self.conv_input = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # 2. Residual Tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # 3. Policy Head — 4x4 board → 16 + 1 pass = 17 actions
        self.p_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.p_bn   = nn.BatchNorm2d(32)
        self.p_fc   = nn.Linear(32 * 4 * 4, 17)

        # 4. Value Head (Categorical: Loss, Draw, Win)
        self.v_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.v_bn   = nn.BatchNorm2d(32)
        self.v_fc1  = nn.Linear(32 * 4 * 4, num_channels)
        self.v_fc2  = nn.Linear(num_channels, 3)

        # 5. Reward Head (Auxiliary)
        self.r_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.r_bn   = nn.BatchNorm2d(32)
        self.r_fc1  = nn.Linear(32 * 4 * 4, num_channels)
        self.r_fc2  = nn.Linear(num_channels, 3)

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # Policy Head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)          # 17 logits

        # Value Head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = self.v_fc2(v)         # 3 logits

        # Reward Head
        r = F.relu(self.r_bn(self.r_conv(x)))
        r = r.view(r.size(0), -1)
        r = F.relu(self.r_fc1(r))
        r = self.r_fc2(r)         # 3 logits

        return p, v, r


# --- DATASET LOADER ---
class TreeDataset(Dataset):
    """
    Binary format per sample (4x4):
      Board(32) + Policy(17) + Mask(17) + V(3) + R(3) + Valid(1) + Weight(1) = 74 floats
    """
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

        # Board(32) + Policy(17) + Mask(17) + V(3) + R(3) + Valid(1) + Weight(1) = 74
        floats_per_sample = 74

        if len(raw) // 4 != self.len * floats_per_sample:
            print(f"Size mismatch. Recalculating...")
            self.len = len(raw) // 4 // floats_per_sample

        print(f"Loading {self.len} samples from {filename}...")
        arr = np.frombuffer(raw, dtype=np.float32).reshape(self.len, floats_per_sample)
        arr = np.nan_to_num(arr, nan=0.0)

        # Slice columns
        self.boards    = arr[:,  0:32].reshape(self.len, 2, 4, 4)
        self.policies  = arr[:, 32:49]
        self.p_masks   = arr[:, 49:66]
        self.target_q  = arr[:, 66:69]
        self.target_r  = arr[:, 69:72]
        self.r_valid   = arr[:, 72]
        self.weights   = arr[:, 73]

        self.print_stats()

    def print_stats(self):
        if self.len == 0: return
        print("\n=== DATASET STATS ===")
        print(f"Total Samples: {self.len}")

        avg_q    = np.mean(self.target_q, axis=0)
        argmax_q = np.argmax(self.target_q, axis=1)
        dist_q   = [np.sum(argmax_q == i) / self.len for i in range(3)]

        print(f"\n[Value Head (Game Outcome)]")
        print(f"  Mean (Soft Avg)   : Loss {avg_q[0]*100:.1f}% | Draw {avg_q[1]*100:.1f}% | Win {avg_q[2]*100:.1f}%")
        print(f"  Argmax (Dominant) : Loss {dist_q[0]*100:.1f}% | Draw {dist_q[1]*100:.1f}% | Win {dist_q[2]*100:.1f}%")

        valid_indices = (self.r_valid > 0.5)
        n_valid = np.sum(valid_indices)
        if n_valid > 0:
            valid_r  = self.target_r[valid_indices]
            avg_r    = np.mean(valid_r, axis=0)
            argmax_r = np.argmax(valid_r, axis=1)
            dist_r   = [np.sum(argmax_r == i) / n_valid for i in range(3)]
            print(f"\n[Reward Head (Terminal Move)] - ({n_valid} samples)")
            print(f"  Mean (Soft Avg)   : Loss {avg_r[0]*100:.1f}% | Draw {avg_r[1]*100:.1f}% | Win {avg_r[2]*100:.1f}%")
            print(f"  Argmax (Dominant) : Loss {dist_r[0]*100:.1f}% | Draw {dist_r[1]*100:.1f}% | Win {dist_r[2]*100:.1f}%")
        else:
            print(f"\n[Reward Head] No terminal states found.")
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


# --- TRAINING ---
def train_epoch(model, opt, loader, epoch):
    model.train()

    stats = {
        "loss_total": 0.0, "loss_p": 0.0, "loss_v": 0.0, "loss_r": 0.0,
        "acc_p_top1": 0.0, "acc_v": 0.0, "acc_r": 0.0,
        "count": 0, "count_r": 0
    }

    dist_v_target = torch.zeros(3).to(DEVICE)
    dist_v_pred   = torch.zeros(3).to(DEVICE)

    for b, tp, pm, tq, tr, rv, w in loader:
        b, tp, pm, tq, tr, rv, w = (
            b.to(DEVICE), tp.to(DEVICE), pm.to(DEVICE),
            tq.to(DEVICE), tr.to(DEVICE), rv.to(DEVICE), w.to(DEVICE)
        )

        w_norm = w / (w.mean() + 1e-9)

        pp, pv, pr = model(b)

        # Policy Loss (masked cross-entropy)
        pp_masked = pp.clone()
        pp_masked[pm == 0] = -1e9
        loss_p_raw = -(tp * F.log_softmax(pp_masked, dim=1)).sum(dim=1)
        loss_p = (loss_p_raw * w_norm).mean()

        # Value Loss
        loss_v_raw = -(tq * F.log_softmax(pv, dim=1)).sum(dim=1)
        loss_v = (loss_v_raw * w_norm).mean()

        # Reward Loss (validity-weighted)
        ce_r = -(tr * F.log_softmax(pr, dim=1)).sum(dim=1)
        valid_weights     = w * rv
        sum_valid_weights = valid_weights.sum() + 1e-6
        loss_r = (ce_r * valid_weights).sum() / sum_valid_weights

        loss = loss_p + loss_v + loss_r

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Batch loss is {loss.item()} (Ignored)")
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        with torch.no_grad():
            batch_n = b.shape[0]
            stats["count"]      += batch_n
            stats["loss_total"] += loss.item() * batch_n
            stats["loss_p"]     += loss_p.item() * batch_n
            stats["loss_v"]     += loss_v.item() * batch_n
            stats["loss_r"]     += loss_r.item() * batch_n

            pred_move   = pp_masked.argmax(dim=1)
            target_move = tp.argmax(dim=1)
            stats["acc_p_top1"] += (pred_move == target_move).sum().item()

            pred_outcome   = pv.argmax(dim=1)
            target_outcome = tq.argmax(dim=1)
            stats["acc_v"] += (pred_outcome == target_outcome).sum().item()

            for i in range(3):
                dist_v_target[i] += (target_outcome == i).sum()
                dist_v_pred[i]   += (pred_outcome   == i).sum()

            valid_mask = (rv > 0.5)
            n_valid    = valid_mask.sum().item()
            if n_valid > 0:
                pred_r   = pr.argmax(dim=1)
                target_r = tr.argmax(dim=1)
                stats["acc_r"]   += (pred_r[valid_mask] == target_r[valid_mask]).sum().item()
                stats["count_r"] += n_valid

    def avg(k, div): return stats[k] / max(stats[div], 1)

    print(f"\n--- Epoch {epoch} Report ---")
    print(f"LOSS    : Total {avg('loss_total','count'):.4f} | Pol {avg('loss_p','count'):.4f} | Val {avg('loss_v','count'):.4f} | Rew {avg('loss_r','count'):.4f}")
    print(f"ACCURACY: Policy {avg('acc_p_top1','count')*100:.1f}%   | Value {avg('acc_v','count')*100:.1f}%  | Reward {avg('acc_r','count_r')*100:.1f}%")

    total = stats["count"]
    dt = (dist_v_target / total * 100).cpu().numpy()
    dp = (dist_v_pred   / total * 100).cpu().numpy()
    print(f"DISTRIB : [Loss / Draw / Win]")
    print(f"  Target: [{dt[0]:4.1f}% {dt[1]:4.1f}% {dt[2]:4.1f}%]")
    print(f"  Pred  : [{dp[0]:4.1f}% {dp[1]:4.1f}% {dp[2]:4.1f}%]")

    if dp[0] > 99 or dp[2] > 99:
        print("  WARNING: Model collapsing to single prediction!")

    return avg('loss_total', 'count')


# --- EXPORT ---
def save_traced(model, filename="model_script.pt"):
    model.cpu().eval()
    example = torch.rand(1, 2, 4, 4)   # 4x4 input

    class ExportWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            p, v, r = self.m(x)
            return p, F.softmax(v, dim=1), F.softmax(r, dim=1)

    traced = torch.jit.trace(ExportWrapper(model), example)
    traced.save(filename)
    model.to(DEVICE)
    print(f"Saved traced model to {filename}")


if __name__ == "__main__":
    if not os.path.exists("model_script.pt"):
        print("Initializing random Othello 4x4 model...")
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
        ld  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(1, 4):
            train_epoch(model, opt, ld, epoch)

        save_traced(model)
        torch.save(model.state_dict(), "weights.pth")
    else:
        print("No data found. Please run C++ engine.")
