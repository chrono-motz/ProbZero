#include "selfplay.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

// --- BINARY SERIALIZER ---
// Saves data as a raw contiguous float array.
// Format: [N_SAMPLES (int32)] [SAMPLE_1 (105 floats)] [SAMPLE_2 (105 floats)] ...
// Per Sample Layout:
//   - Board: 84 floats
//   - Policy: 7 floats
//   - Mask: 7 floats
//   - Value Target: 3 floats
//   - Reward Target: 3 floats
//   - Reward Valid: 1 float
void save_dataset_binary(const std::vector<DataFormat>& dataset, const std::string& filename) {
    if (dataset.empty()) {
        std::cerr << "Warning: Dataset is empty. Nothing to save.\n";
        return;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    // 1. Write Header (Number of Samples)
    int32_t n_samples = static_cast<int32_t>(dataset.size());
    out.write(reinterpret_cast<const char*>(&n_samples), sizeof(int32_t));

    // 2. Write Body
    // We buffer the writes to avoid thousands of tiny syscalls
    // 50 + 26 + 26 + 3 + 3 + 1 = 226 floats * 4 bytes = 1060 bytes per sample.
    for (const auto& sample : dataset) {
        // Board (50)
        out.write(reinterpret_cast<const char*>(sample.board.data()), 50 * sizeof(float));
        
        // Policy (26)
        out.write(reinterpret_cast<const char*>(sample.policy.data()), 26 * sizeof(float));
        
        // Mask (26)
        out.write(reinterpret_cast<const char*>(sample.policy_mask.data()), 26 * sizeof(float));
        
        // Value Target (3)
        out.write(reinterpret_cast<const char*>(sample.target_q.data()), 3 * sizeof(float));
        
        // Reward Target (3)
        out.write(reinterpret_cast<const char*>(sample.target_r.data()), 3 * sizeof(float));
        
        // Reward Valid (1)
        out.write(reinterpret_cast<const char*>(&sample.reward_valid), sizeof(float));

        // Weight (1)
        out.write(reinterpret_cast<const char*>(&sample.weight), sizeof(float));
    }

    out.close();
    std::cout << "Successfully saved " << n_samples << " samples to " << filename << " (" 
              << (n_samples * 1064 / 1024 / 1024) << " MB)\n";
}

int main(int argc, char* argv[]) {
    // Configuration
    std::string model_path = "model_script.pt"; // Ensure this exists!
    std::string output_file = "dataset.bin";
    int batch_size = 8;  // Particles per batch (Flow Volume)
    int num_batches = 60; // Total expansion steps (Tree Size ~= batch_size * num_batches)
    int num_games = 20;   // Number of parallel games

    // Optional: Parse args
    if (argc > 1) model_path = argv[1];
    if (argc > 2) output_file = argv[2];
    if (argc > 3) num_batches = std::stoi(argv[3]);
    if (argc > 4) num_games = std::stoi(argv[4]);

    std::cout << "=== Connect4 AlphaZero Tree Harvester ===\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Target Batches: " << num_batches << "\n";
    std::cout << "Particles/Batch: " << batch_size << "\n";
    std::cout << "Parallel Games: " << num_games << "\n";
    
    try {
        // 1. Initialize Engine
        Selfplay engine(model_path, num_games);

        // 2. Run Generation
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<DataFormat> dataset;
        engine.generate_tree_data(batch_size, num_batches, dataset);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // 3. Stats
        std::cout << "----------------------------------------\n";
        std::cout << "Time Elapsed: " << elapsed.count() << "s\n";
        std::cout << "Positions Generated: " << dataset.size() << "\n";
        std::cout << "Speed: " << (dataset.size() / elapsed.count()) << " positions/sec\n";
        
        // 4. Save
        save_dataset_binary(dataset, output_file);

    } catch (const std::exception& e) {

        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
















// #include "selfplay.hpp"
// #include <iostream>
// #include <string>
// #include <algorithm>

// // --- HELPER: Coordinate Converter ---
// int parse_move(std::string s) {
//     if (s == "PASS" || s == "pass") return 64;
//     if (s.length() < 2) return -1;
//     char col = std::toupper(s[0]);
//     char row = s[1];
//     if (col < 'A' || col > 'H') return -1;
//     if (row < '1' || row > '8') return -1;
//     return (row - '1') * 8 + (col - 'A');
// }

// std::string format_move(int m) {
//     if (m == 64) return "PASS";
//     char col = 'A' + (m % 8);
//     char row = '1' + (m / 8);
//     return std::string(1, col) + std::string(1, row);
// }

// // --- HELPER: Visualizer ---
// // IN src/main.cpp

// void print_board(const Othello& game) {
//     float buf[50];
//     game.encode_state(buf); // [0..63] is Self, [64..127] is Opponent
    
//     int p = game.current_player(); // 0 = Black, 1 = White
    
//     std::cout << "\n   A B C D E F G H\n";
//     for (int r = 0; r < 8; ++r) {
//         std::cout << (r + 1) << " ";
//         for (int c = 0; c < 8; ++c) {
//             int idx = r * 8 + c;
            
//             // LOGIC FIX: Map "Self/Opp" back to "Black/White" for display
//             bool is_self = (buf[idx] == 1.0f);
//             bool is_opp  = (buf[64 + idx] == 1.0f);
            
//             if (p == 0) { 
//                 // Black's Turn: Self=Black(X), Opp=White(O)
//                 if (is_self) std::cout << " X";
//                 else if (is_opp) std::cout << " O";
//                 else std::cout << " .";
//             } else { 
//                 // White's Turn: Self=White(O), Opp=Black(X)
//                 if (is_self) std::cout << " O";     // Self is White!
//                 else if (is_opp) std::cout << " X"; // Opp is Black!
//                 else std::cout << " .";
//             }
//         }
//         std::cout << "\n";
//     }
//     std::cout << "Turn: " << (p == 0 ? "Black (X)" : "White (O)") << "\n";
// }

// int main(int argc, char* argv[]) {
//     std::cout << "=== Othello Arena ===\n";
//     std::cout << "Select Mode:\n";
//     std::cout << "1. Human (Black) vs AI (White)\n";
//     std::cout << "2. AI (Black) vs Human (White)\n";
//     std::cout << "3. AI vs AI (Model Battle)\n";
    
//     int mode;
//     if (!(std::cin >> mode)) return 0;

//     std::string model1_path = "model_script.pt";
//     std::string model2_path = "model_script.pt";

//     if (mode == 3) {
//         std::cout << "Enter Path for Black Bot: ";
//         std::cin >> model1_path;
//         std::cout << "Enter Path for White Bot: ";
//         std::cin >> model2_path;
//     }

//     Mcts bot1; 
//     Mcts bot2;
//     Othello game;

//     std::cout << "Loading Models...\n";
//     bot1.load_model(model1_path);
//     bot1.initialize(game);

//     bot2.load_model(model2_path);
//     bot2.initialize(game);

//     // MCTS SETTINGS
//     int total_sims = 800; 
//     int batch_size = 16;
//     int num_batches = total_sims / batch_size;

//     while (!game.is_terminal()) {
//         print_board(game);
//         int current_p = game.current_player(); 
//         int move = -1;

//         // --- HUMAN INPUT CHECK ---
//         bool is_human = false;
//         if (mode == 1 && current_p == 0) is_human = true;
//         if (mode == 2 && current_p == 1) is_human = true;

//         if (is_human) {
//              std::string s;
//              std::cout << "Your Move (" << (current_p==0?"Black":"White") << "): ";
//              std::cin >> s;
//              move = parse_move(s);
//         } else {
//             // --- AI LOGIC ---
//             // std::cout << "Bot thinking...\n";
//             Mcts* active_bot = (current_p == 0) ? &bot1 : &bot2;
            
//             // Loop batches to allow tree expansion
//             for(int b=0; b<num_batches; ++b) {
//                 active_bot->run_simulation_batch(batch_size);
//             }
            
//             move = active_bot->get_move(true);
//         }

//         // Apply Move
//         if (move != -1) {
//             std::cout << "Played: " << format_move(move) << "\n";
//             game.apply_action(move);
            
//             // Sync Helper
//             auto sync_bot = [&](Mcts& bot) {
//                 // If bot has the move, advance. Else reset.
//                 // Note: You must have moved root_idx to public in mcts.hpp for this!
//                 int r = bot.root_idx;
//                 if (r >= 0 && r < bot.nodes.size() && bot.nodes[r].children[move] != -1) {
//                     bot.advance_root(move);
//                 } else {
//                     bot.initialize(game);
//                 }
//             };

//             sync_bot(bot1);
//             sync_bot(bot2);

//         } else {
//             std::cout << "Invalid Move! (Bot returned -1 or Bad Input)\n";
//             if (!is_human) break; // Stop infinite loop if bot fails
//         }
//     }

//     print_board(game);
//     std::cout << "GAME OVER\n";
//     std::cout << "Black Win: " << game.check_win(0) << "\n";
//     std::cout << "White Win: " << game.check_win(1) << "\n";

//     return 0;
// }