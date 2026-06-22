#include "selfplay.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

// --- BINARY SERIALIZER ---
void save_dataset_binary(const std::vector<DataFormat>& dataset, const std::string& filename, int board_size) {
    if (dataset.empty()) {
        std::cerr << "Warning: Dataset is empty. Nothing to save.\n";
        return;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    int32_t n_samples = static_cast<int32_t>(dataset.size());
    out.write(reinterpret_cast<const char*>(&n_samples), sizeof(int32_t));

    int num_actions = board_size * board_size + 1;
    int board_dim = 2 * board_size * board_size;
    int total_floats = board_dim + 2 * num_actions + 8;

    for (const auto& sample : dataset) {
        out.write(reinterpret_cast<const char*>(sample.board.data()),       board_dim * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.policy.data()),      num_actions * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.policy_mask.data()), num_actions * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.target_q.data()),     3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.target_r.data()),     3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(&sample.reward_valid),           sizeof(float));
        out.write(reinterpret_cast<const char*>(&sample.weight),                 sizeof(float));
    }

    out.close();
    std::cout << "Successfully saved " << n_samples << " samples to " << filename
              << " (" << (n_samples * total_floats * 4 / 1024 / 1024) << " MB)\n";
}

// --- HELPER: Visuals ---
void print_board(const Othello& game) {
    int board_size = game.get_board_size();
    int board_dim = 2 * board_size * board_size;
    std::vector<float> buf(board_dim);
    game.encode_state(buf.data());
    int p = game.current_player();

    std::cout << "\n  ";
    for (int c = 0; c < board_size; ++c) {
        std::cout << " " << char('A' + c);
    }
    std::cout << "\n";
    
    for (int r = 0; r < board_size; ++r) {
        std::cout << (r + 1) << " ";
        for (int c = 0; c < board_size; ++c) {
            int idx = r * board_size + c;
            bool is_black, is_white;
            if (p == 0) {
                is_black = (buf[idx]      == 1.0f);
                is_white = (buf[board_size*board_size + idx] == 1.0f);
            } else {
                is_white = (buf[idx]      == 1.0f);
                is_black = (buf[board_size*board_size + idx] == 1.0f);
            }
            if      (is_black) std::cout << " X";
            else if (is_white) std::cout << " O";
            else               std::cout << " .";
        }
        std::cout << "\n";
    }
    std::cout << "Turn: " << (p == 0 ? "Black (X)" : "White (O)") << "\n";
}

int parse_move(std::string s, int board_size) {
    if (s == "PASS" || s == "pass") return board_size * board_size;
    if (s.length() < 2) return -1;
    char col = std::toupper(s[0]);
    char row = s[1];
    if (col < 'A' || col >= 'A' + board_size) return -1;
    if (row < '1' || row >= '1' + board_size) return -1;
    return (row - '1') * board_size + (col - 'A');
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
    int board_size;

    struct Match {
        Othello game;
        std::unique_ptr<Mcts> ai_b;
        std::unique_ptr<Mcts> ai_w;
        bool finished = false;
        int winner = -1;
        Match(int bs) : game(bs) {}
    };

    std::vector<std::unique_ptr<Match>> matches;

    void run_batch_inference(std::vector<Mcts*>& batch, std::shared_ptr<Model>& model) {
        if (batch.empty()) return;
        int total_leaves = 0;
        for (auto* ai : batch) total_leaves += ai->get_leaf_count();
        if (total_leaves == 0) return;

        int board_dim = 2 * board_size * board_size;
        std::vector<float> input_buffer(total_leaves * board_dim);
        float* ptr = input_buffer.data();
        for (auto* ai : batch) {
            ai->extract_leaf_states(ptr);
            ptr += ai->get_leaf_count() * board_dim;
        }

        auto result_tuple = model->inference(input_buffer, total_leaves, board_size);
        auto& p_tens = std::get<0>(result_tuple);
        auto& v_tens = std::get<1>(result_tuple);
        auto& r_tens = std::get<2>(result_tuple);

        const float* p_ptr = p_tens.data_ptr<float>();
        const float* v_ptr = v_tens.data_ptr<float>();
        const float* r_ptr = r_tens.data_ptr<float>();

        int offset = 0;
        for (auto* ai : batch) {
            int c = ai->get_leaf_count();
            if (c > 0) {
                ai->update_leaves_from_tensor(
                    p_ptr + offset * (board_size * board_size + 1),
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
    ParallelArena(std::string path_b, std::string path_w, int n, int s, int bs) : num_games(n), sims(s), board_size(bs) {
        model_b = std::make_shared<Model>(); model_b->load_model(path_b);
        model_w = std::make_shared<Model>(); model_w->load_model(path_w);

        matches.resize(n);
        for (int i = 0; i < n; ++i) {
            matches[i] = std::make_unique<Match>(board_size);
            matches[i]->ai_b = std::make_unique<Mcts>(board_size); matches[i]->ai_b->set_model(model_b);
            matches[i]->ai_w = std::make_unique<Mcts>(board_size); matches[i]->ai_w->set_model(model_w);
            matches[i]->ai_b->initialize(matches[i]->game);
            matches[i]->ai_w->initialize(matches[i]->game);
        }
    }

    void run() {
        int finished_count = 0;
        std::cout << "Running " << num_games << " games in parallel (" << sims << " sims)...\n";

        int iter = 0;
        while (finished_count < num_games) {
            iter++;
            if (iter % 10 == 0)
                std::cout << "Processed turn " << iter << " | Completed: " << finished_count << "\n";

            int sim_step = 16;
            for (int s = 0; s < sims; s += sim_step) {
                std::vector<Mcts*> batch_b, batch_w;
                for (auto& m : matches) {
                    if (m->finished) continue;
                    int p = m->game.current_player();
                    if (p == 0) batch_b.push_back(m->ai_b.get());
                    else        batch_w.push_back(m->ai_w.get());
                }
                for (auto* ai : batch_b) ai->run_simulation_prepare(sim_step);
                for (auto* ai : batch_w) ai->run_simulation_prepare(sim_step);
                run_batch_inference(batch_b, model_b);
                run_batch_inference(batch_w, model_w);
            }

            for (auto& m : matches) {
                if (m->finished) continue;
                int p = m->game.current_player();
                Mcts* ai  = (p == 0) ? m->ai_b.get() : m->ai_w.get();
                Mcts* opp = (p == 0) ? m->ai_w.get() : m->ai_b.get();

                int move = ai->get_move(true);
                if (move >= 0) {
                    m->game.apply_action(move);
                    ai->advance_root(move);
                    opp->advance_root(move);
                } else {
                    if (!m->game.is_terminal()) {
                        std::vector<int> legal;
                        if (m->game.get_legal_actions(legal) > 0) {
                            m->game.apply_action(legal[0]);
                            ai->advance_root(legal[0]);
                            opp->advance_root(legal[0]);
                        }
                    }
                }

                if (m->game.is_terminal()) {
                    m->finished = true;
                    m->winner = get_winner(m->game);
                    finished_count++;
                }
            }
        }

        int wins_b = 0, wins_w = 0, draws = 0;
        for (auto& m : matches) {
            if      (m->winner == 0) wins_b++;
            else if (m->winner == 1) wins_w++;
            else                     draws++;
        }
        std::cout << "Arena Result: Black " << wins_b << " | White " << wins_w << " | Draws " << draws << "\n";
    }
};

// --- MAIN ---
int main(int argc, char* argv[]) {
    int mode = 1;
    std::string path1 = "model_b.pt";
    std::string path2 = "model_w.pt";
    int num_games = 1;
    int board_size = 4; // default

    // Parse CLI arguments: --mode, --path1, --path2, --games, --size
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = std::stoi(argv[++i]);
        else if (arg == "--path1" && i + 1 < argc) path1 = argv[++i];
        else if (arg == "--path2" && i + 1 < argc) path2 = argv[++i];
        else if (arg == "--games" && i + 1 < argc) num_games = std::stoi(argv[++i]);
        else if (arg == "--size" && i + 1 < argc) board_size = std::stoi(argv[++i]);
    }

    if (mode == 4 && num_games > 1) {
        ParallelArena arena(path1, path2, num_games, 1600, board_size);
        arena.run();
        return 0;
    }

    // Selfplay data generation mode
    if (mode == 0) {
        Selfplay selfplay(path1, board_size, num_games, 0);
        std::vector<DataFormat> dataset;
        selfplay.generate_tree_data(400, 10, dataset); 
        save_dataset_binary(dataset, path2, board_size);
        return 0;
    }

    std::cout << "=== OTHELLO " << board_size << "x" << board_size << " ARENA ===\n";
    std::cout << "Mode: " << mode << "\n";

    Othello game(board_size);
    std::vector<Mcts*> ai_players(2, nullptr);

    try {
        if (mode == 1) {
            ai_players[1] = new Mcts(board_size);
            ai_players[1]->load_model(path1);
            ai_players[1]->initialize(game);
            std::cout << "White AI loaded: " << path1 << "\n";
        } else if (mode == 2 || mode == 3) {
            ai_players[0] = new Mcts(board_size);
            ai_players[0]->load_model(path1);
            ai_players[0]->initialize(game);
            std::cout << "Black AI loaded: " << path1 << "\n";
        } else if (mode == 4) {
            ai_players[0] = new Mcts(board_size);
            ai_players[0]->load_model(path1);
            ai_players[0]->initialize(game);
            ai_players[1] = new Mcts(board_size);
            ai_players[1]->load_model(path2);
            ai_players[1]->initialize(game);
            std::cout << "AI vs AI | Black: " << path1 << " | White: " << path2 << "\n";
        }
    } catch (...) {
        std::cerr << "Critical error loading models!\n";
        return 1;
    }

    int ai_sims = 1200;

    while (!game.is_terminal()) {
        if (num_games == 1 && mode != 5) print_board(game);
        int p = game.current_player();
        int move = -1;

        if (ai_players[p] != nullptr) {
            Mcts* current_ai = ai_players[p];
            for (int i = 0; i < (ai_sims / 16); ++i) current_ai->run_simulation_batch(16);
            move = current_ai->get_move(true);

            if (num_games == 1 && move >= 0) {
                const auto& node = current_ai->nodes[current_ai->root_idx];

                std::cout << "\n=== Search Evaluation ===\n";
                std::cout << "Loss : " << node.value_search[0] * 100.0f << "%\n";
                std::cout << "Draw : " << node.value_search[1] * 100.0f << "%\n";
                std::cout << "Win  : " << node.value_search[2] * 100.0f << "%\n";

                std::cout << "\n=== Value Evaluation ===\n";
                std::cout << "Loss : " << node.value_net[0] * 100.0f << "%\n";
                std::cout << "Draw : " << node.value_net[1] * 100.0f << "%\n";
                std::cout << "Win  : " << node.value_net[2] * 100.0f << "%\n";

                std::cout << "\n=== Outcome Evaluation ===\n";
                std::cout << "Loss : " << node.reward_net[0] * 100.0f << "%\n";
                std::cout << "Draw : " << node.reward_net[1] * 100.0f << "%\n";
                std::cout << "Win  : " << node.reward_net[2] * 100.0f << "%\n";

                std::cout << "\nProb Head Probabilities\n";

                for (int move : node.legal) {
                    std::cout
                        << format_move(move, board_size)
                        << " : "
                        << node.policy_net[move] * 100.0f
                        << "%\n";
                }

                std::cout << "\nAfter Search Probabilities\n";

                for (int move : node.legal) {
                    std::cout
                        << format_move(move, board_size)
                        << " : "
                        << node.policy_search[move] * 100.0f
                        << "%\n";
                }
            }
        } else if (mode == 3 && p == 1) {
            std::vector<int> legal;
            int n = game.get_legal_actions(legal);
            move = legal[rand() % n];
        } else {
            std::vector<int> legal;
            int n_legal = game.get_legal_actions(legal);
            if (n_legal == 1 && legal[0] == board_size * board_size) {
                std::cout << "Forced Pass.\n";
                move = board_size * board_size;
            } else {
                while (true) {
                    std::cout << "Enter Move (e.g., B2) or PASS: ";
                    std::string s; std::cin >> s;
                    move = parse_move(s, board_size);
                    bool is_legal = false;
                    for (int i = 0; i < n_legal; ++i) if (legal[i] == move) is_legal = true;
                    if (is_legal) break;
                    std::cout << "Invalid move.\n";
                }
            }
        }

        if (move == -1) {
            std::vector<int> legal;
            int n = game.get_legal_actions(legal);
            if (n > 0) move = legal[0];
            else break;
        }

        std::cout << ">>> PLAYED: " << format_move(move, board_size) << "\n";
        game.apply_action(move);

        for (int i = 0; i < 2; ++i) {
            if (ai_players[i] != nullptr) {
                int r = ai_players[i]->root_idx;
                if (r >= 0 && ai_players[i]->nodes[r].children[move] != -1) {
                    ai_players[i]->advance_root(move);
                } else {
                    ai_players[i]->initialize(game);
                }
            }
        }
    }

    print_board(game);
    std::cout << "=== GAME OVER ===\n";
    int winner = get_winner(game);
    if      (winner == 0) std::cout << "Black Winner!\n";
    else if (winner == 1) std::cout << "White Winner!\n";
    else                  std::cout << "Draw.\n";

    if (ai_players[0]) delete ai_players[0];
    if (ai_players[1]) delete ai_players[1];

    return 0;
}
