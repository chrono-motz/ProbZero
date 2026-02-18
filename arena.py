import subprocess
import re
import math
import random
from collections import defaultdict

# --- CONFIGURATION ---
ENGINE_PATH = "/home/chrono_motz/SmolEngine/build/play" # Adjusted path
MODELS = [
    "best_model_1.pt", # Adjusted paths (local)
    "best_model_2.pt",
]
GAMES_PER_PAIR = 20  # Total games between each pair
K_FACTOR = 32        # How much Elo changes per game
INITIAL_ELO = 1500
TEMP = 0.5

# --- ELO LOGIC ---
def get_expected_score(res_a, res_b):
    return 1 / (1 + 10 ** ((res_b - res_a) / 400))

# --- ENGINE WRAPPER ---
def run_match_parallel(path_black, path_white, num_games):
    """Runs the C++ binary in parallel mode."""
    cmd = [ENGINE_PATH, "4", path_black, path_white, str(num_games), str(TEMP)]
    
    try:
        print(f"Running match: {path_black} vs {path_white} ({num_games} games)...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stderr:
            print(f"[STDERR] {result.stderr[:500]}...") # Print first 500 chars to avoid spam

        output = result.stdout
        
        # Parse: Arena Result: Black X | White Y | Draws Z
        m = re.search(r"Arena Result: Black (\d+) \| White (\d+) \| Draws (\d+)", output)
        
        if m:
            b = int(m.group(1))
            w = int(m.group(2))
            d = int(m.group(3))
            return b, w, d
        else:
            print(f"Warning: Could not parse output. Output:\n{output}")
            return 0, 0, 0
    except Exception as e:
        print(f"Error running match: {e}")
        return 0, 0, 0

# --- ARENA MAIN ---
def main():
    # Helper to find models
    models_found = []
    for m in MODELS:
        if "__" in m: continue # Check existing
        models_found.append(m) # Assume existence or handle errors
    
    # Just use configured list for now
    
    elo_ratings = {m: INITIAL_ELO for m in MODELS}
    stats = {m: {"wins": 0, "losses": 0, "draws": 0} for m in MODELS}
    
    pairs = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            pairs.append((MODELS[i], MODELS[j]))

    print(f"Starting Tournament: {len(MODELS)} models, {len(pairs) * GAMES_PER_PAIR} total games.\n")

    for m1, m2 in pairs:
        # Part 1: m1 Black, m2 White
        n_black = GAMES_PER_PAIR // 2
        b, w, d = run_match_parallel(m1, m2, n_black)
        
        # Update Elo m1 (Black) vs m2 (White)
        exp_m1 = get_expected_score(elo_ratings[m1], elo_ratings[m2])
        actual_score_m1 = b + 0.5 * d
        
        elo_ratings[m1] += K_FACTOR * (actual_score_m1 - exp_m1 * n_black)
        elo_ratings[m2] += K_FACTOR * ((n_black - actual_score_m1) - (1 - exp_m1) * n_black)
        
        stats[m1]["wins"] += b
        stats[m1]["losses"] += w
        stats[m1]["draws"] += d
        stats[m2]["wins"] += w
        stats[m2]["losses"] += b
        stats[m2]["draws"] += d
        
        # Part 2: m2 Black, m1 White
        n_white = GAMES_PER_PAIR - n_black
        b2, w2, d2 = run_match_parallel(m2, m1, n_white)
        
        # Update Elo m2 (Black) vs m1 (White)
        exp_m2 = get_expected_score(elo_ratings[m2], elo_ratings[m1])
        actual_score_m2 = b2 + 0.5 * d2
        
        elo_ratings[m2] += K_FACTOR * (actual_score_m2 - exp_m2 * n_white)
        elo_ratings[m1] += K_FACTOR * ((n_white - actual_score_m2) - (1 - exp_m2) * n_white)
        
        stats[m2]["wins"] += b2
        stats[m2]["losses"] += w2
        stats[m2]["draws"] += d2
        stats[m1]["wins"] += w2
        stats[m1]["losses"] += b2
        stats[m1]["draws"] += d2

        print(f"Finished Matchup: {m1} vs {m2} -> {b+w2}-{w+b2}-{d+d2}")

    # --- RESULTS ---
    print("\n" + "="*50)
    print(f"{'Model':<25} | {'Elo':<6} | {'W-L-D':<10}")
    print("-" * 50)
    
    sorted_models = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    for model, elo in sorted_models:
        s = stats[model]
        wld = f"{s['wins']}-{s['losses']}-{s['draws']}"
        name = model.split('/')[-1]
        print(f"{name:<25} | {int(elo):<6} | {wld:<10}")
    print("="*50)


if __name__ == "__main__":
    main()