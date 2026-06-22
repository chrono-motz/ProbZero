#include "mcts.hpp"
#include "model.hpp"
#include "node.hpp"
#include <optional>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

std::string format_move(int m, int board_size) {
    if (m == board_size * board_size) return "PASS";
    char col = 'A' + (m % board_size);
    char row = '1' + (m / board_size);
    return std::string(1, col) + std::string(1, row);
}

Mcts::Mcts(int size) : model(make_shared<Model>()), model_opp(make_shared<Model>()), board_size(size) {
    num_actions = board_size * board_size + 1;
    board_dim = 2 * board_size * board_size;
    
    // Defer initialization of nodes so we don't copy the initial node without the right board_size.
    // Instead we use ptrs or just initialize when needed. 
    // We will reserve and resize.
    nodes.reserve(200000); 
    traversal_set.resize(200000, 0);
    terminal_set.resize(200000, 0);
    traversal_epoch = 0;
    terminal_epoch = 0;
}

Mcts::~Mcts() = default;

bool Mcts::is_game_over() {
    return nodes[root_idx].state.is_terminal();
}

void Mcts::load_model(const std::string& path) {
    model->load_model(path);
}

void Mcts::load_model(const std::string& path, const std::string& path_opp) {
    model->load_model(path);
    model_opp->load_model(path_opp);
}

int Mcts::get_move(bool deterministic) {
    if (root_idx < 0 || root_idx >= static_cast<int>(nodes.size())) return -1;
    
    const Node& root = nodes[root_idx];
    if (root.num_legal < 0 || root.num_legal > num_actions) return -1;

    if (deterministic) {
        int best_move = -1;
        float max_p = -1.0f;
        for (int i = 0; i < root.num_legal; ++i) {
            int m = root.legal[i];
            if (m < 0 || m >= num_actions) continue;
            if (root.children[m] != -1) {
                float p = root.policy_search[m];
                if (p > max_p) {
                    max_p = p;
                    best_move = m;
                }
            }
        }
        return best_move;
    } else {
        std::vector<float> probs;
        std::vector<int> moves;
        probs.reserve(root.num_legal);
        moves.reserve(root.num_legal);

        for (int i = 0; i < root.num_legal; ++i) {
            int m = root.legal[i];
            if (m < 0 || m >= num_actions) continue;
            if (root.children[m] != -1) {
                probs.push_back(root.policy_search[m]);
                moves.push_back(m);
            }
        }

        if (probs.empty()) {
            if (root.num_legal > 0) return root.legal[0];
            return -1;
        }

        static thread_local std::mt19937 rng(std::random_device{}());
        float epsilon = 0;
        std::uniform_real_distribution<float> unif(0.0f, 1.0f);
        
        int chosen_index;
        if (unif(rng) < epsilon) {
            std::uniform_int_distribution<int> rand_dist(0, moves.size() - 1);
            chosen_index = rand_dist(rng);
        } else {
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            chosen_index = dist(rng);
        }
        return moves[chosen_index];
    }
}

void Mcts::advance_root(int move_idx) {
    if (root_idx < 0 || root_idx >= static_cast<int>(nodes.size())) return;
    int child_idx = nodes[root_idx].children[move_idx];
    if (child_idx < 0 || child_idx >= static_cast<int>(nodes.size())) return;
    root_idx = child_idx;
}

void Mcts::initialize(const Othello& game) {
    nodes.clear();
    // Using emplace_back to instantiate with the correct board size
    nodes.emplace_back(board_size); 
    nodes.resize(200000, Node(board_size)); 
    leaves.clear();
    node_map.clear();

    nodes[0].state = game;
    nodes[0].num_legal = game.get_legal_actions(nodes[0].legal);

    root_idx = 0;
    node_count_idx = 1;
    traversal_epoch++;
    terminal_epoch++;
}

void Mcts::run_simulation_batch(int volume) {
    leaves.clear();
    traversal_epoch++;
    forward(root_idx, volume);
    process_batch(leaves);
    for (int leaf_idx : leaves) {
        if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) continue;
        mark_traversal_path(leaf_idx);
        if (nodes[leaf_idx].terminal) {
            mark_terminal_path(leaf_idx);
        }
    } 
    forward_update(0); 
}

void Mcts::run_simulation_prepare(int volume) {
    leaves.clear();
    traversal_epoch++;
    forward(root_idx, volume);
}

void Mcts::extract_leaf_states(float* buffer) {
    int batch_size = leaves.size();
    for (int i = 0; i < batch_size; ++i) {
        int leaf_idx = leaves[i];
        if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) continue;
        nodes[leaf_idx].state.encode_state(buffer + (i * board_dim));
    }
}

void Mcts::update_leaves_from_tensor(const float* p_data_base, const float* v_data_base, const float* r_data_base, int batch_size) {
    if (batch_size != static_cast<int>(leaves.size())) return;
    
    for (int i = 0; i < batch_size; ++i) {
        int idx = leaves[i];
        if (idx < 0 || idx >= static_cast<int>(nodes.size())) continue;
        Node& node = nodes[idx];

        if (node.state.is_terminal()) {
            node.terminal = true;
            if (node.state.check_win(node.state.current_player())){
                node.value_net = {0.0f , 0.0f , 1.0f};
                node.value_search = {0.0f , 0.0f , 1.0f};
                node.value_target = {0.0f , 0.0f , 1.0f};
                node.reward_net = {0.0f , 0.0f , 1.0f};
                node.reward_target = {0.0f , 0.0f , 1.0f};
            }else if (node.state.check_win(1-node.state.current_player())){
                node.value_net = {1.0f , 0.0f , 0.0f};
                node.value_search = {1.0f , 0.0f , 0.0f};
                node.value_target = {1.0f , 0.0f , 0.0f};
                node.reward_net = {1.0f , 0.0f , 0.0f};
                node.reward_target = {1.0f , 0.0f , 0.0f};
            }else{
                node.value_net = {0.0f , 1.0f , 0.0f};
                node.value_search = {0.0f , 1.0f , 0.0f};
                node.value_target = {0.0f , 1.0f , 0.0f};
                node.reward_net = {0.0f , 1.0f , 0.0f};
                node.reward_target = {0.0f , 1.0f , 0.0f};
            }
        } else {
             std::array<float, 3> value_net;
             std::array<float, 3> reward_net;
             for(int k=0; k<3; ++k) value_net[k] = v_data_base[i*3 + k];
             for(int k=0; k<3; ++k) reward_net[k] = r_data_base[i*3 + k];

             node.value_net = value_net; 
             node.value_search = value_net;
             node.value_target = reward_net;
             node.reward_net = reward_net;

             float max_logit = -1e9f;
             for(int m=0; m < node.num_legal; ++m) {
                int move = node.legal[m];
                float logit = p_data_base[i*num_actions + move];
                if (std::isnan(logit)) logit = -1e9f;
                if (logit > max_logit) max_logit = logit;
             }
             
             float sum_p = 0.0f;
             std::vector<float> exp_p(node.num_legal);
             
             for(int m=0; m < node.num_legal; ++m) {
                int move = node.legal[m];
                float logit = p_data_base[i*num_actions + move];
                if (std::isnan(logit)) logit = -1e9f;
                exp_p[m] = std::exp(logit - max_logit); 
                sum_p += exp_p[m];
             }

            if (sum_p < 1e-9f) sum_p = 1e-9f; 
            float inv_sum = 1.0f / sum_p;
            for (int m = 0; m < node.num_legal; ++m){
                int move = node.legal[m];
                node.policy_net[move] = exp_p[m]*inv_sum;
                node.policy_search[move] = node.policy_net[move];
            }
            node.explored = true;
        }
    }
}

void Mcts::run_simulation_finalize() {
    for (int leaf_idx : leaves){
        if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) continue;
        mark_traversal_path(leaf_idx);
        if (nodes[leaf_idx].terminal) {
            mark_terminal_path(leaf_idx);
        }
    } 
    forward_update(0); 
}

void Mcts::final_update(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    if (terminal_set[node_idx] == terminal_epoch) {
        terminal_set[node_idx] = 0;
    } else {
        return;
    }

    Node& node = nodes[node_idx];
    if (node.terminal) return;

    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];
        int c_idx = node.children[m];
        if (c_idx != -1 && terminal_set[c_idx] == terminal_epoch) {
            final_update(c_idx);
        }
    }

    std::vector<int> active_children;
    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];
        int c_idx = node.children[m];
        if (c_idx != -1) active_children.push_back(c_idx);
    }

    int n_active = active_children.size();
    if (n_active == 0) return;

    std::vector<float> weights(n_active);
    winner_prob(n_active, active_children, weights, param_temperature);
    
    node.reward_target = {0.0f, 0.0f, 0.0f};
    
    for (int i = 0; i < n_active; ++i) {
        int c_idx = active_children[i];
        float w = weights[i];
        const auto& cr = nodes[c_idx].reward_target;
        node.reward_target[0] += w * cr[2];
        node.reward_target[1] += w * cr[1];
        node.reward_target[2] += w * cr[0];
    }

    float sum = node.reward_target[0] + node.reward_target[1] + node.reward_target[2];
    if (sum > 1e-9f) {
        float inv = 1.0f / sum;
        node.reward_target[0] *= inv;
        node.reward_target[1] *= inv;
        node.reward_target[2] *= inv;
    }
}

void Mcts::winner_prob(int num_children, const std::vector<int>& children, std::vector<float>& weights, float temp_scale) {
    std::fill(weights.begin(), weights.end(), 0.0f);
    if (num_children == 0) return;
    std::vector<float> p0(num_children), p1(num_children), p2(num_children), lt1(num_children), lt2(num_children);
    
    for (int i = 0; i < num_children; ++i) {
        int c_idx = children[i];
        if (c_idx < 0 || c_idx >= static_cast<int>(nodes.size())) {
             p0[i] = 0.0f; p1[i] = 0.0f; p2[i] = 0.0f;
             lt1[i] = 1e-12f; lt2[i] = 1e-12f;
             continue;
        }
        const auto& q = nodes[c_idx].value_search;
        p0[i] = q[2]; 
        p1[i] = q[1]; 
        p2[i] = q[0]; 
        lt1[i] = p0[i] + 1e-12f;
        lt2[i] = p0[i] + p1[i] + 1e-12f;
    }

    double prod_lt1 = 1.0;
    double prod_lt2 = 1.0;
    for (int i = 0; i < num_children; ++i) {
        prod_lt1 *= lt1[i];
        prod_lt2 *= lt2[i];
    }

    float sum_win = 0.0f;
    std::vector<float> raw_win(num_children);
    for (int i = 0; i < num_children; ++i) {
        float term1 = p1[i] * static_cast<float>(prod_lt1 / lt1[i]);
        float term2 = p2[i] * static_cast<float>(prod_lt2 / lt2[i]);
        raw_win[i] = term1 + term2;
        sum_win += raw_win[i];
    }

    if (sum_win < 1e-9f) {
        float uniform = 1.0f / num_children;
        for (int i = 0; i < num_children; ++i) weights[i] = uniform;
    } else {
        if (std::abs(temp_scale - 1.0f) > 1e-6f) {
             float sum_pow = 0.0f;
             if (std::abs(temp_scale) < 1e-9f) {
                 float inv_t = 1000.0f; 
                 for(int i=0; i<num_children; ++i) {
                     raw_win[i] = std::pow(raw_win[i], inv_t);
                     sum_pow += raw_win[i];
                 }
                 sum_win = sum_pow;
             } else {
                 float inv_t = 1.0f / temp_scale;
                 for(int i=0; i<num_children; ++i) {
                     raw_win[i] = std::pow(raw_win[i], inv_t);
                     sum_pow += raw_win[i];
                 }
                 sum_win = sum_pow;
             }
        }

        if (sum_win < 1e-9f) {
             float uniform = 1.0f / num_children;
             for (int i = 0; i < num_children; ++i) weights[i] = uniform;
        } else {
             float inv_sum = 1.0f / sum_win;
             for (int i = 0; i < num_children; ++i) weights[i] = raw_win[i] * inv_sum;
        }
    }
}

void Mcts::forward(int node_idx, int volume) {
    std::vector<int> leaves_threaded;
    leaves_threaded.reserve(volume);
    forward_recursive(node_idx, volume, leaves_threaded);
    if (!leaves_threaded.empty()) {
        leaves.insert(leaves.end(), leaves_threaded.begin(), leaves_threaded.end());
    }
}

void Mcts::process_batch(const std::vector<int>& leaves_local) {
    if (leaves_local.empty()) return;
    int batch_size = leaves_local.size();
    
    std::vector<float> input_buffer(batch_size * board_dim);
    for (int i = 0; i < batch_size; ++i) {
        int leaf_idx = leaves_local[i];
        nodes[leaf_idx].state.encode_state(&input_buffer[i * board_dim]);
    }

    auto result_tuple = model->inference(input_buffer, batch_size, board_size);
    auto& p_tens = std::get<0>(result_tuple);
    auto& v_tens = std::get<1>(result_tuple);
    auto& r_tens = std::get<2>(result_tuple);

    const float* p_acc = p_tens.data_ptr<float>();
    const float* v_acc = v_tens.data_ptr<float>();
    const float* r_acc = r_tens.data_ptr<float>();

    for (int i = 0; i < batch_size; ++i) {
        int idx = leaves_local[i];
        Node& node = nodes[idx];
        
        if (node.state.is_terminal()) {
            node.terminal = true;
            if (node.state.check_win(node.state.current_player())){
                node.value_net = {0.0f , 0.0f , 1.0f};
                node.value_search = {0.0f , 0.0f , 1.0f};
                node.value_target = {0.0f , 0.0f , 1.0f};
                node.reward_net = {0.0f , 0.0f , 1.0f};
                node.reward_target = {0.0f , 0.0f , 1.0f};
            }else if (node.state.check_win(1-node.state.current_player())){
                node.value_net = {1.0f , 0.0f , 0.0f};
                node.value_search = {1.0f , 0.0f , 0.0f};
                node.value_target = {1.0f , 0.0f , 0.0f};
                node.reward_net = {1.0f , 0.0f , 0.0f};
                node.reward_target = {1.0f , 0.0f , 0.0f};
            }else{
                node.value_net = {0.0f , 1.0f , 0.0f};
                node.value_search = {0.0f , 1.0f , 0.0f};
                node.value_target = {0.0f , 1.0f , 0.0f};
                node.reward_net = {0.0f , 1.0f , 0.0f};
                node.reward_target = {0.0f , 1.0f , 0.0f};
            }
        }else {
            if (std::isnan(v_acc[i*3 + 0]) || std::isnan(r_acc[i*3 + 0])) {
                node.value_net = {0.0f, 0.0f, 0.0f};
                node.reward_net = {0.0f, 0.0f, 0.0f};
            } else {
                std::array<float, 3> value_net = { v_acc[i*3 + 0], v_acc[i*3 + 1], v_acc[i*3 + 2] };
                std::array<float, 3> reward_net = { r_acc[i*3 + 0], r_acc[i*3 + 1], r_acc[i*3 + 2] };
                node.value_net = value_net; 
                node.value_search = value_net;
                node.value_target = value_net; 
                node.reward_net = reward_net;
            }
        } 
        
        float max_logit = -1e9f;
        for(int m=0; m < node.num_legal; ++m) {
            float logit = p_acc[i*num_actions + node.legal[m]]; 
            if (std::isnan(logit)) logit = -1e9f;
            if (logit > max_logit) max_logit = logit;
        }

        float sum_p = 0.0f;
        std::vector<float> exp_p(node.num_legal);
        for(int m=0; m < node.num_legal; ++m) {
            float logit = p_acc[i*num_actions + node.legal[m]];
            if (std::isnan(logit)) logit = -1e9f;
            exp_p[m] = std::exp(logit - max_logit); 
            sum_p += exp_p[m];
        }

        if (sum_p < 1e-9f) sum_p = 1e-9f; 
        float inv_sum = 1.0f / sum_p;
        for (int m = 0; m < node.num_legal; ++m){
            node.policy_net[node.legal[m]] = exp_p[m]*inv_sum;
            node.policy_search[node.legal[m]] = node.policy_net[node.legal[m]];
        }
        node.explored = true;
    }
}

void Mcts::mark_traversal_path(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    if (traversal_set[node_idx] == traversal_epoch) return;
    traversal_set[node_idx] = traversal_epoch;
    Node& node = nodes[node_idx];
    for (int p : node.parents) {
        if (p >= 0 && p < static_cast<int>(nodes.size())) {
            mark_traversal_path(p);
        }
    }
}

void Mcts::mark_terminal_path(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    if (terminal_set[node_idx] == terminal_epoch) return;
    terminal_set[node_idx] = terminal_epoch;
    Node& node = nodes[node_idx];
    for (int p : node.parents) {
        if (p >= 0 && p < static_cast<int>(nodes.size())) {
            mark_terminal_path(p);
        }
    }
}

void Mcts::forward_recursive(int node_idx, int volume, std::vector<int>& leaves_threaded) {
    if (volume == 0) return;
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    
    Node* node = &nodes[node_idx];
    node->visit_count += volume;
    
    if (!node->explored){
        leaves_threaded.push_back(node_idx);
    } else if (node->terminal){
        return;
    } else {
        std::vector<float> probs;
        std::vector<int> prob_index_to_move; 
        probs.reserve(node->num_legal + 1);
        prob_index_to_move.reserve(node->num_legal);

        std::vector<std::pair<float, int>> unexplored;
        float bucket_sum = 0.0f;

        for (int i = 0; i < node->num_legal; ++i) {
            int m = node->legal[i];
            int c_idx = node->children[m];
            float p = node->policy_search[m];

            if (c_idx != -1) {
                if (!nodes[c_idx].terminal) {
                    probs.push_back(p);
                    prob_index_to_move.push_back(m);
                }
            } else {
                bucket_sum += p;
                unexplored.push_back({p, m});
            }
        }

        bool has_bucket = !unexplored.empty();
        int bucket_idx = -1;
        if (has_bucket) {
            probs.push_back(bucket_sum);
            bucket_idx = probs.size() - 1; 
        }

        if (probs.empty()) return;

        static thread_local std::mt19937 rng(std::random_device{}());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        
        std::vector<int> counts(probs.size(), 0);
        for (int i = 0; i < volume; ++i) {
            counts[dist(rng)]++;
        }

        for (int i = 0; i < static_cast<int>(prob_index_to_move.size()); ++i) {
            if (counts[i] > 0) {
                int move = prob_index_to_move[i];
                int c_idx = node->children[move];
                forward_recursive(c_idx, counts[i], leaves_threaded);
            }
        }

        if (has_bucket && counts[bucket_idx] > 0) {
            std::sort(unexplored.begin(), unexplored.end(), std::greater<std::pair<float, int>>());
            int particles = counts[bucket_idx];
            for (auto& p : unexplored) {
                if (particles <= 0) break;
                int move = p.second;
                Othello next_state = node->state;
                next_state.apply_action(move);
                uint64_t key = next_state.key();
                int child_idx = -1;
                
                auto it = node_map.find(key);
                if (it != node_map.end()) {
                    child_idx = it->second;
                    auto& parents = nodes[child_idx].parents;
                    if (std::find(parents.begin(), parents.end(), node_idx) == parents.end()) {
                        parents.push_back(node_idx);
                    }
                } else {
                    if (node_count_idx >= static_cast<int>(nodes.size())) break; 
                    child_idx = node_count_idx++;
                    nodes[child_idx] = Node(board_size); 
                    nodes[child_idx].state = next_state;
                    nodes[child_idx].parents.push_back(node_idx);
                    nodes[child_idx].num_legal = next_state.get_legal_actions(nodes[child_idx].legal);
                    nodes[child_idx].terminal = next_state.is_terminal(); 
                    node_map[key] = child_idx;
                }
                node->children[move] = child_idx;
                forward_recursive(child_idx, 1, leaves_threaded);
                particles--;
            }
        }
    }
}

void Mcts::forward_update(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) return;
    if (traversal_set[node_idx] != traversal_epoch) return;
    traversal_set[node_idx] = 0; 

    Node& node = nodes[node_idx];
    if (node.terminal) return;

    std::vector<int> active_child_idx;
    std::vector<int> active_move_idx;
    float explored_p = 0.0f;

    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];
        int c_idx = node.children[m];
        if (c_idx != -1) {
            if (traversal_set[c_idx] == traversal_epoch) {
                forward_update(c_idx); 
                active_child_idx.push_back(c_idx);
                active_move_idx.push_back(m);
                explored_p += node.policy_search[m];
            }
        }
    }

    int n_active = active_child_idx.size();
    if (n_active == 0) return;

    std::vector<float> scaled_probs(n_active);
    winner_prob(n_active, active_child_idx, scaled_probs, param_temperature);

    for (int i = 0; i < n_active; ++i) {
        node.policy_search[active_move_idx[i]] = scaled_probs[i] * explored_p;
    }

    float remaining_p = std::max(0.0f, 1.0f - explored_p);
    std::array<float, 3> new_v_search = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> new_v_target = {0.0f, 0.0f, 0.0f};

    for (int i = 0; i < n_active; ++i) {
        float w = scaled_probs[i] * explored_p;
        int c_idx = active_child_idx[i];
        const auto& cv_search = nodes[c_idx].value_search;
        const auto& cv_target = nodes[c_idx].value_target;

        new_v_search[0] += w * cv_search[2]; 
        new_v_search[1] += w * cv_search[1]; 
        new_v_search[2] += w * cv_search[0]; 

        new_v_target[0] += w * cv_target[2];
        new_v_target[1] += w * cv_target[1];
        new_v_target[2] += w * cv_target[0];
    }

    new_v_search[0] += remaining_p * node.value_net[0];
    new_v_search[1] += remaining_p * node.value_net[1];
    new_v_search[2] += remaining_p * node.value_net[2];

    new_v_target[0] += remaining_p * node.reward_net[0];
    new_v_target[1] += remaining_p * node.reward_net[1];
    new_v_target[2] += remaining_p * node.reward_net[2];

    node.value_search = new_v_search;
    node.value_target = new_v_target;
}

void Mcts::print_debug_root(int limit_moves) {
    if (root_idx < 0 || root_idx >= static_cast<int>(nodes.size())) return;
    const Node& root = nodes[root_idx];
    
    std::cout << "\n[DEBUG ROOT] Node Idx: " << root_idx << " | Legal Moves: " << root.num_legal << "\n";
    std::cout << ">>> 1. Raw NN Probs (Policy Net): ";
    for(int i=0; i< std::min(root.num_legal, limit_moves); ++i) {
        int m = root.legal[i];
        std::cout << format_move(m, board_size) << "=" << root.policy_net[m] << " ";
    }
    std::cout << "...\n";

    std::vector<int> active_child_idx;
    std::vector<int> active_move_idx;
    for (int i = 0; i < root.num_legal; ++i) {
        int m = root.legal[i];
        int c_idx = root.children[m];
        if (c_idx != -1) {
            active_child_idx.push_back(c_idx);
            active_move_idx.push_back(m);
        }
    }

    int n_active = active_child_idx.size();
    if (n_active == 0) return;

    std::vector<float> unscaled_weights(n_active);
    winner_prob(n_active, active_child_idx, unscaled_weights, 1.0f);
    std::cout << ">>> 2. Search Probs (Temp=1.0): ";
    for (int i = 0; i < std::min(n_active, limit_moves); ++i) {
        std::cout << format_move(active_move_idx[i], board_size) << "=" << unscaled_weights[i] << " ";
    }
    std::cout << "...\n";

    std::vector<float> scaled_weights(n_active);
    winner_prob(n_active, active_child_idx, scaled_weights, param_temperature);
    std::cout << ">>> 3. Search Probs (Temp=" << param_temperature << "): ";
    for (int i = 0; i < std::min(n_active, limit_moves); ++i) {
        std::cout << format_move(active_move_idx[i], board_size) << "=" << scaled_weights[i] << " ";
    }
    std::cout << "...\n\n";
}
