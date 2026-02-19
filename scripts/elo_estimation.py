"""
ProbZero Elo Estimation Script (C++ Arena)
==========================================
Uses the pre-built C++ 'play' binary for GPU-accelerated MCTS arena matches.
Computes Elo ratings and generates publication-ready plots.

Usage (from repository root):
    python scripts/elo_estimation.py
"""

import subprocess
import re
import os
import itertools
import sys
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# CONFIG — adjust these paths for your setup
# =============================================
# Assume script is in subfolder (e.g., /scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root is one level up
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

PLAY_BINARY = os.path.join(REPO_ROOT, "build", "play")
# Models are now in /models folder
MODEL_DIR = os.path.join(REPO_ROOT, "models")
# Results saved to /results folder
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

GAMES_PER_MATCHUP = 20     # Total games per matchup (half as each color)


# =============================================
# ARENA RUNNER (uses C++ binary)
# =============================================
def run_arena_match(model_a_path, model_b_path, num_games):
    """
    Run C++ arena: model_a plays Black, model_b plays White.
    Returns (black_wins, white_wins, draws).
    """
    cmd = [
        PLAY_BINARY,
        "4",                      # mode 4 = parallel arena
        model_a_path,
        model_b_path,
        str(num_games),
    ]
    
    # Set LD_LIBRARY_PATH for libtorch if needed (optional)
    env = os.environ.copy()
    # libtorch_lib = os.path.join(REPO_ROOT, "libtorch", "lib")
    # if os.path.exists(libtorch_lib):
    #     env["LD_LIBRARY_PATH"] = libtorch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env
        )
        output = result.stdout + result.stderr
        
        # Parse: "Arena Result: Black X | White Y | Draws Z"
        match = re.search(r'Arena Result:\s*Black\s+(\d+)\s*\|\s*White\s+(\d+)\s*\|\s*Draws\s+(\d+)', output)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            print(f"  WARNING: Could not parse output:\n{output[-300:]}")
            return 0, 0, 0
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Match timed out")
        return 0, 0, 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0, 0, 0


def run_full_matchup(model_a_path, model_b_path, num_games):
    """
    Run a full matchup: half games with A as Black, half with A as White.
    Returns (wins_a, wins_b, draws).
    """
    half = num_games // 2
    
    # A = Black, B = White
    bw_b, bw_w, bw_d = run_arena_match(model_a_path, model_b_path, half)
    wins_a_1 = bw_b      # A was Black, Black wins = A wins
    wins_b_1 = bw_w      # B was White, White wins = B wins
    draws_1 = bw_d
    
    # B = Black, A = White
    bw_b2, bw_w2, bw_d2 = run_arena_match(model_b_path, model_a_path, num_games - half)
    wins_b_2 = bw_b2     # B was Black, Black wins = B wins
    wins_a_2 = bw_w2     # A was White, White wins = A wins  
    draws_2 = bw_d2
    
    return (wins_a_1 + wins_a_2, wins_b_1 + wins_b_2, draws_1 + draws_2)


# =============================================
# ELO CALCULATION
# =============================================
def compute_elo_ratings(results, k=32, initial_elo=1000, iterations=50):
    """
    Compute Elo ratings from match results.
    results: list of (iter_a, iter_b, wins_a, wins_b, draws)
    """
    elos = defaultdict(lambda: initial_elo)
    
    for _ in range(iterations):
        for a, b, wa, wb, draws in results:
            total = wa + wb + draws
            if total == 0:
                continue
            score_a = (wa + 0.5 * draws) / total
            score_b = (wb + 0.5 * draws) / total
            ea = 1.0 / (1.0 + 10 ** ((elos[b] - elos[a]) / 400.0))
            eb = 1.0 - ea
            elos[a] += k * (score_a - ea)
            elos[b] += k * (score_b - eb)
    
    # Normalize: lowest = 0
    min_elo = min(elos.values())
    for key in elos:
        elos[key] -= min_elo
    
    return dict(elos)


# =============================================
# PLOTTING
# =============================================
def create_plots(sorted_elos, results, iters, output_dir):
    """Generate publication-quality Elo and win-rate plots."""
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'figure.facecolor': 'white'
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Plot 1: Elo Progression ---
    ax1 = axes[0]
    plot_iters = [it for it, _ in sorted_elos]
    plot_elos = [elo for _, elo in sorted_elos]
    
    ax1.plot(plot_iters, plot_elos, 'o-', color='#2196F3', linewidth=2.5, markersize=9,
             markerfacecolor='white', markeredgecolor='#2196F3', markeredgewidth=2.5,
             zorder=3)
    ax1.fill_between(plot_iters, plot_elos, alpha=0.12, color='#2196F3')
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Elo Rating')
    ax1.set_title('ProbZero: Elo Rating vs Training Iteration', fontweight='bold')
    ax1.grid(True, alpha=0.25, linestyle='--')
    
    if len(plot_iters) > 0:
        ax1.set_xlim(min(plot_iters) - 15, max(plot_iters) + 15)
        ax1.set_ylim(min(plot_elos) - 50, max(plot_elos) + 80)
    
    for it, elo in sorted_elos:
        ax1.annotate(f'{elo:.0f}', (it, elo), textcoords="offset points",
                     xytext=(0, 14), ha='center', fontsize=9, fontweight='bold',
                     color='#1565C0')
    
    # --- Plot 2: Win Rate Heatmap ---
    ax2 = axes[1]
    n = len(iters)
    wm = np.full((n, n), 0.5)
    
    for a, b, wa, wb, d in results:
        try:
            i, j = iters.index(a), iters.index(b)
            total = wa + wb + d
            if total > 0:
                wm[i][j] = (wa + 0.5 * d) / total
                wm[j][i] = (wb + 0.5 * d) / total
        except ValueError:
            continue # Should not happen if iters aligned
    
    im = ax2.imshow(wm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([str(it) for it in iters], fontsize=9)
    ax2.set_yticklabels([str(it) for it in iters], fontsize=9)
    ax2.set_xlabel('Opponent Iteration')
    ax2.set_ylabel('Player Iteration')
    ax2.set_title('Win Rate Matrix (Row vs Column)', fontweight='bold')
    
    for i in range(n):
        for j in range(n):
            v = wm[i][j]
            color = 'white' if abs(v - 0.5) > 0.3 else 'black'
            ax2.text(j, i, f'{v:.0%}', ha='center', va='center',
                     fontsize=8, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax2, label='Win Rate', shrink=0.8)
    
    plt.tight_layout(pad=2.0)
    
    plot_path = os.path.join(output_dir, 'elo_progression.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    # plt.show() # Not needed in headless
    print(f"\nPlot saved to {plot_path}")


# =============================================
# MAIN
# =============================================
def main():
    # Verify binary exists
    if not os.path.exists(PLAY_BINARY):
        print(f"ERROR: Play binary not found at {PLAY_BINARY}")
        print("Please build with: cd build && cmake .. && make play")
        return
    
    # Find models
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found at {MODEL_DIR}")
        return
    
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')])
    iters = []
    for f in model_files:
        try:
            # model_iter_XXX.pt
            it = int(f.replace('model_iter_', '').replace('.pt', ''))
            iters.append(it)
        except ValueError:
            continue
            
    iters = sorted(iters)
    
    print(f"ProbZero Elo Estimation")
    print(f"{'='*60}")
    print(f"Binary:        {PLAY_BINARY}")
    print(f"Models:        {MODEL_DIR}")
    print(f"Found models:  {iters}")
    print(f"Games/matchup: {GAMES_PER_MATCHUP}")
    print(f"{'='*60}\n")
    
    if len(iters) < 2:
        print("Not enough models to run arena (need at least 2).")
        return

    # Build model paths
    model_paths = {it: os.path.join(MODEL_DIR, f"model_iter_{it}.pt") for it in iters}
    
    # Run all-pairs matchups
    pairs = list(itertools.combinations(iters, 2))
    results = []
    
    print(f"Running {len(pairs)} matchups...\n")
    
    for idx, (a, b) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] iter_{a} vs iter_{b}:", end=" ")
        wa, wb, d = run_full_matchup(
            model_paths[a], model_paths[b], GAMES_PER_MATCHUP
        )
        results.append((a, b, wa, wb, d))
        print(f"{wa}-{wb}-{d}")
    
    # Compute Elo
    elos = compute_elo_ratings(results)
    sorted_elos = sorted(elos.items(), key=lambda x: x[0])
    
    print(f"\n{'='*60}")
    print(f"ELO RATINGS")
    print(f"{'='*60}")
    print(f"\n{'Iteration':>10s} | {'Elo':>8s}")
    print(f"{'-'*25}")
    for it, elo in sorted_elos:
        print(f"{'iter_'+str(it):>10s} | {elo:>8.1f}")
    
    # Match results table
    print(f"\n{'='*60}")
    print(f"MATCH RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Matchup':>25s} | {'W-L-D':>10s} | {'Win%':>6s}")
    print(f"{'-'*50}")
    for a, b, wa, wb, d in results:
        total = wa + wb + d
        wp = (wa + 0.5*d) / total * 100 if total > 0 else 50
        print(f"  iter_{a:>3d} vs iter_{b:>3d} | {wa:>3d}-{wb:>3d}-{d:>3d} | {wp:>5.1f}%")
    
    # Generate plots
    create_plots(sorted_elos, results, iters, OUTPUT_DIR)
    
    # Save raw data
    results_path = os.path.join(OUTPUT_DIR, 'elo_results.txt')
    with open(results_path, 'w') as f:
        f.write("ProbZero Elo Estimation Results\n")
        f.write(f"Games/matchup: {GAMES_PER_MATCHUP}\n\n")
        f.write("Elo Ratings:\n")
        for it, elo in sorted_elos:
            f.write(f"  iter_{it}: {elo:.1f}\n")
        f.write("\nMatch Results:\n")
        for a, b, wa, wb, d in results:
            f.write(f"  iter_{a} vs iter_{b}: {wa}-{wb}-{d}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
