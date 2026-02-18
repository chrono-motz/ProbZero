#include "game.hpp"
#include "mcts.hpp"
#include "node.hpp"
#include "model.hpp"
#include <iostream>
#include <string>
#include <algorithm>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <atomic>

// --- HELPER: Visuals ---
void print_board(const Othello& game) {
    float buf[50];
    game.encode_state(buf); 
    int p = game.current_player(); 
    
    std::cout << "\n   A B C D E\n";
    for (int r = 0; r < 5; ++r) {
        std::cout << (r + 1) << " ";
        for (int c = 0; c < 5; ++c) {
            int idx = r * 5 + c;
            bool is_black, is_white;
            if (p == 0) { // Black's view
                is_black = (buf[idx] == 1.0f);
                is_white = (buf[25 + idx] == 1.0f);
            } else { // White's view
                is_white = (buf[idx] == 1.0f);
                is_black = (buf[25 + idx] == 1.0f);
            }

            if (is_black) std::cout << " X";      
            else if (is_white) std::cout << " O"; 
            else std::cout << " .";
        }
        std::cout << "\n";
    }
    std::cout << "Turn: " << (p == 0 ? "Black (X)" : "White (O)") << "\n";
}

int parse_move(std::string s) {
    if (s == "PASS" || s == "pass") return 64;
    if (s.length() < 2) return -1;
    char col = std::toupper(s[0]);
    char row = s[1];
    if (col < 'A' || col > 'E') return -1;
    if (row < '1' || row > '5') return -1;
    return (row - '1') * 5 + (col - 'A');
}

std::string format_move(int m) {
    if (m == 64) return "PASS";
    char col = 'A' + (m % 5);
    char row = '1' + (m / 5);
    return std::string(1, col) + std::string(1, row);
}

int get_winner(const Othello& game) {
    if (game.check_win(0)) return 0;
    if (game.check_win(1)) return 1;
    return 2; // Draw
}

class ParallelArena {
    std::shared_ptr<Model> model_b;
    std::shared_ptr<Model> model_w;
    int num_games;
    int sims;
    
    struct Match {
        Othello game;
        std::unique_ptr<Mcts> ai_b;
        std::unique_ptr<Mcts> ai_w;
        bool finished = false;
        int winner = -1; // 0: Black, 1: White, 2: Draw
    };
    
    std::vector<Match> matches;

    void run_batch_inference(std::vector<Mcts*>& batch, std::shared_ptr<Model>& model) {
        if (batch.empty()) return;
        
        int total_leaves = 0;
        for(auto* ai : batch) total_leaves += ai->get_leaf_count();
        if (total_leaves == 0) return;
        
        std::vector<float> input_buffer(total_leaves * 50);
        float* ptr = input_buffer.data();
        for(auto* ai : batch) {
            ai->extract_leaf_states(ptr);
            ptr += ai->get_leaf_count() * 50;
        }
        
        auto result_tuple = model->inference(input_buffer, total_leaves);
        auto& p_tens = std::get<0>(result_tuple);
        auto& v_tens = std::get<1>(result_tuple);
        auto& r_tens = std::get<2>(result_tuple);
        
        const float* p_ptr = p_tens.data_ptr<float>();
        const float* v_ptr = v_tens.data_ptr<float>();
        const float* r_ptr = r_tens.data_ptr<float>();
        
        int offset = 0;
        for(auto* ai : batch) {
            int c = ai->get_leaf_count();
            if (c > 0) {
                ai->update_leaves_from_tensor(
                    p_ptr + offset * 26,
                    v_ptr + offset * 3,
                    r_ptr + offset * 3,
                    c
                );
                ai->run_simulation_finalize();
                offset += c;
            }
        }
    }

public:
    ParallelArena(std::string path_b, std::string path_w, int n, int s, float temp) : num_games(n), sims(s) {
        model_b = std::make_shared<Model>(); model_b->load_model(path_b);
        model_w = std::make_shared<Model>(); model_w->load_model(path_w);
        
        matches.resize(n);
        for(int i=0; i<n; ++i) {
            matches[i].ai_b = std::make_unique<Mcts>(); 
            matches[i].ai_b->set_model(model_b);
            matches[i].ai_b->set_temperature(temp); // Set temperature
            
            matches[i].ai_w = std::make_unique<Mcts>(); 
            matches[i].ai_w->set_model(model_w);
            matches[i].ai_w->set_temperature(temp); // Set temperature

            matches[i].ai_b->initialize(matches[i].game);
            matches[i].ai_w->initialize(matches[i].game);
        }
    }
    
    void run() {
        int finished_count = 0;
        std::cout << "Running " << num_games << " games in parallel (" << sims << " sims)...\n";
        
        int iter = 0;
        while(finished_count < num_games) {
            iter++;
            if(iter % 10 == 0) std::cout << "Processed turn " << iter << " | Completed: " << finished_count << "\n";

            // 1. MCTS SEARCH PHASE
            int sim_step = 16;
            for(int s=0; s < sims; s += sim_step) { 
                std::vector<Mcts*> batch_b;
                std::vector<Mcts*> batch_w;
                
                for(auto& m : matches) {
                    if(m.finished) continue;
                    int p = m.game.current_player();
                    if(p == 0) batch_b.push_back(m.ai_b.get());
                    else       batch_w.push_back(m.ai_w.get());
                }
                
                // Prepare
                for(auto* ai : batch_b) ai->run_simulation_prepare(sim_step);
                for(auto* ai : batch_w) ai->run_simulation_prepare(sim_step);
                
                // Inference
                run_batch_inference(batch_b, model_b);
                run_batch_inference(batch_w, model_w);
            }
            
            // 2. PLAY PHASE
            for(auto& m : matches) {
                if(m.finished) continue;
                
                int p = m.game.current_player();
                Mcts* ai = (p==0) ? m.ai_b.get() : m.ai_w.get();
                Mcts* opp = (p==0) ? m.ai_w.get() : m.ai_b.get();
                
                int move = ai->get_move(true); // Deterministic
                // Apply move if valid
                if(move >= 0) {
                   m.game.apply_action(move);
                   ai->advance_root(move);
                   // Opponent's tree never explored current player's moves,
                   // so reinitialize from the updated game state
                   opp->initialize(m.game);
                } else {
                   // Pass?
                   if (!m.game.is_terminal()) {
                        std::array<int, 26> legal; 
                        if (m.game.get_legal_actions(legal) > 0) {
                             m.game.apply_action(legal[0]);
                             ai->advance_root(legal[0]);
                             opp->initialize(m.game);
                        }
                   }
                }

                if(m.game.is_terminal()) {
                     m.finished = true;
                     m.winner = get_winner(m.game);
                     finished_count++;
                }
            }
        }
        
        // Print Stats
        int wins_b = 0, wins_w = 0, draws = 0;
        for(auto& m : matches) {
            if(m.winner == 0) wins_b++;
            else if(m.winner == 1) wins_w++;
            else draws++;
        }
        std::cout << "Arena Result: Black " << wins_b << " | White " << wins_w << " | Draws " << draws << "\n";
    }
};

// --- MAIN ARENA ---
int main(int argc, char* argv[]) {
    /* Usage: ...
       4: AI(B) [path_1] vs AI(W) [path_2] [num_games]
    */
    
    int mode = 1;
    std::string path1 = "model_b.pt";
    std::string path2 = "model_w.pt";
    int num_games = 1;
    float temperature = 1.0f;
    
    if (argc > 1) mode = std::stoi(argv[1]);
    if (argc > 2) path1 = argv[2];
    if (argc > 3) path2 = argv[3];
    if (argc > 4) num_games = std::stoi(argv[4]);
    if (argc > 5) temperature = std::stof(argv[5]);

    if(mode == 4 && num_games > 1) {
        ParallelArena arena(path1, path2, num_games, 1600, temperature);
        arena.run();
        return 0;
    }

    std::cout << "=== OTHELLO ARENA ===\n";
    std::cout << "Mode: " << mode << "\n";

    Othello game;
    std::vector<Mcts*> ai_players(2, nullptr);

    try {
        if (mode == 1) {
            ai_players[1] = new Mcts();
            ai_players[1]->set_temperature(temperature);
            ai_players[1]->load_model(path1);
            ai_players[1]->initialize(game);
            std::cout << "White AI loaded: " << path1 << "\n";
        } 
        else if (mode == 2 || mode == 3) {
            ai_players[0] = new Mcts();
            ai_players[0]->set_temperature(temperature);
            ai_players[0]->load_model(path1);
            ai_players[0]->initialize(game);
            std::cout << "Black AI loaded: " << path1 << "\n";
        } 
        else if (mode == 4) {
            ai_players[0] = new Mcts();
            ai_players[0]->set_temperature(temperature);
            ai_players[0]->load_model(path1);
            ai_players[0]->initialize(game);
            
            ai_players[1] = new Mcts();
            ai_players[1]->set_temperature(temperature);
            ai_players[1]->load_model(path2);
            ai_players[1]->initialize(game);
            std::cout << "AI vs AI Match Started!\n";
            std::cout << "Black: " << path1 << " | White: " << path2 << "\n";
        }
    } catch (...) {
        std::cerr << "Critical error loading models!\n";
        return 1;
    }

    int ai_sims = 1600;

    while (!game.is_terminal()) {
        if (num_games == 1 && mode != 5) print_board(game);
        int p = game.current_player();
        int move = -1;

        if (ai_players[p] != nullptr) {
            Mcts* current_ai = ai_players[p];
            for(int i=0; i < (ai_sims/16); ++i) current_ai->run_simulation_batch(16);
            
            static int _debug_ctr = 0;
            if (_debug_ctr < 5) { // Print first 5 moves only
                 current_ai->print_debug_root(10);
                 _debug_ctr++;
            }
            
            move = current_ai->get_move(true); 
            
            if (num_games==1 && move >= 0) {
                const auto& node = current_ai->nodes[current_ai->root_idx];
                std::cout << "Confidence: " << int(node.value_search[2] * 100) << "%\n";
            }
        } 
        else if (mode == 3 && p == 1) {
            std::array<int, 26> legal;
            int n = game.get_legal_actions(legal);
            move = legal[rand() % n];
        } 
        else {
            std::array<int, 26> legal;
            int n_legal = game.get_legal_actions(legal);
            if (n_legal == 1 && legal[0] == 64) {
                std::cout << "Forced Pass.\n";
                move = 64;
            } else {
                while (true) {
                    std::cout << "Enter Move (e.g., C3): ";
                    std::string s; std::cin >> s;
                    move = parse_move(s);
                    bool is_legal = false;
                    for(int i=0; i<n_legal; ++i) if(legal[i] == move) is_legal = true;
                    if (is_legal) break;
                    std::cout << "Invalid move.\n";
                }
            }
        }
        
        // --- APPLY MOVE AND UPDATE ALL AI TREES ---
        if (move == -1) {
             std::array<int, 26> legal;
             int n = game.get_legal_actions(legal);
             if (n > 0) move = legal[0];
             else break; 
        }

        std::cout << ">>> PLAYED: " << format_move(move) << "\n";
        game.apply_action(move);

        for (int i = 0; i < 2; ++i) {
            if (ai_players[i] != nullptr) {
                // If the move exists in the tree, move root; else reset.
                ai_players[i]->advance_root(move);
            }
        }
    }
    
    // --- GAME OVER ---
    print_board(game);
    std::cout << "=== GAME OVER ===\n";
    int winner = get_winner(game);
    if (winner == 0) std::cout << "Black Winner? 1\n";
    else if (winner == 1) std::cout << "White Winner? 1\n";
    else std::cout << "Draw.\n";

    // Clean up
    if(ai_players[0]) delete ai_players[0];
    if(ai_players[1]) delete ai_players[1];

    return 0;
}