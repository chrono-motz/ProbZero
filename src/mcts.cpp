#include "mcts.hpp"
#include "model.hpp"
#include "node.hpp"
#include <optional>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

Mcts :: Mcts() : model(make_shared<Model>()) , model_opp(make_shared<Model>()){
    nodes.resize(200000);  // Large fixed allocation
    traversal_set.resize(200000,0);
    terminal_set.resize(200000,0);
    traversal_epoch = 0;
    terminal_epoch = 0;
}

Mcts::~Mcts() = default ;

bool Mcts::is_game_over(){
    return nodes[root_idx].state.is_terminal();
}
void Mcts::load_model(const std::string& path){
    model->load_model(path);
}
void Mcts::load_model(const std::string& path , const std::string& path_opp){
    model->load_model(path);
    model_opp->load_model(path_opp);
}
int Mcts::get_move(bool deterministic) {
    // 1. Get the root node
    if (root_idx < 0 || root_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid root_idx " << root_idx << "\n";
        return -1;
    }
    
    const Node& root = nodes[root_idx];
    
    // Sanity check on num_legal
    if (root.num_legal < 0 || root.num_legal > 26) {
        std::cerr << "ERROR: Invalid num_legal " << root.num_legal << " at root_idx " << root_idx << "\n";
        return -1;
    }

    if (deterministic) {
        // COMPETITIVE PLAY / EVALUATION
        // Return the move with the highest probability among EXPLORED moves
        int best_move = -1;
        float max_p = -1.0f;

        for (int i = 0; i < root.num_legal; ++i) {
            int m = root.legal[i];
            // Bounds check on the move
            if (m < 0 || m >= 26) {
                std::cerr << "ERROR: Invalid move " << m << " at index " << i << "\n";
                continue;
            }
            // Only consider moves that have been explored (have children)
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
        // SELF-PLAY TRAINING
        // Sample from policy_search distribution (only explored moves)
        
        // Gather probabilities for EXPLORED legal moves
        std::vector<float> probs;
        std::vector<int> moves;
        probs.reserve(root.num_legal);
        moves.reserve(root.num_legal);

        for (int i = 0; i < root.num_legal; ++i) {
            int m = root.legal[i];
            // Bounds check on the move
            if (m < 0 || m >= 26) {
                std::cerr << "ERROR: Invalid move " << m << " at index " << i << "\n";
                continue;
            }
            // Only include moves that have children
            if (root.children[m] != -1) {
                probs.push_back(root.policy_search[m]);
                moves.push_back(m);
            }
        }

        if (probs.empty()) {
            std::cerr << "ERROR: No explored moves available at root_idx " << root_idx << "\n";
            // Fallback: return first legal move even if not explored
            if (root.num_legal > 0) {
                return root.legal[0];
            }
            return -1;
        }

        // Create distribution and sample
        static thread_local std::mt19937 rng(std::random_device{}());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        
        int chosen_index = dist(rng);
        return moves[chosen_index];
    }
}
void Mcts::advance_root(int move_idx){
    if (root_idx < 0 || root_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid root_idx in advance_root: " << root_idx << "\n";
        return;
    }
    int child_idx = nodes[root_idx].children[move_idx];
    if (child_idx < 0 || child_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid child_idx from move " << move_idx << ": " << child_idx << "\n";
        return;
    }
    root_idx = child_idx;
}
void Mcts::initialize(const Othello& game){
    nodes.clear();
    nodes.resize(200000);  // Large fixed allocation
    leaves.clear();
    node_map.clear();

    nodes[0].state = game;
    nodes[0].num_legal = game.get_legal_actions(nodes[0].legal);


    root_idx = 0;
    node_count_idx = 1;
    traversal_epoch ++;
    terminal_epoch ++;
}
void Mcts::run_simulation_batch(int volume){
    leaves.clear();
    traversal_epoch++;
    forward(root_idx, volume);
    process_batch(leaves);
    for (int leaf_idx : leaves){
        if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) {
            std::cerr << "ERROR: Invalid leaf_idx " << leaf_idx << " in run_simulation_batch\n";
            continue;
        }
        mark_traversal_path(leaf_idx);
        if (nodes[leaf_idx].terminal) {
            mark_terminal_path(leaf_idx);
        }
    } 
    forward_update(0); 
}
void Mcts::run_simulation_prepare(int volume){
    leaves.clear();
    traversal_epoch++;
    forward(root_idx, volume);
}

void Mcts::extract_leaf_states(float* buffer){
    int batch_size = leaves.size();
    for (int i = 0; i < batch_size; ++i) {
        int leaf_idx = leaves[i];
         if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) continue;
        nodes[leaf_idx].state.encode_state(buffer + (i * 50));
    }
}

void Mcts::update_leaves_from_tensor(const float* p_data_base, const float* v_data_base, const float* r_data_base, int batch_size){
    if (batch_size != leaves.size()) {
        std::cerr << "Mismatch in update_leaves: " << batch_size << " vs " << leaves.size() << "\n";
        return;
    }
    
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
                float logit = p_data_base[i*26 + move];
                if (std::isnan(logit)) logit = -1e9f;
                if (logit > max_logit) max_logit = logit;
             }
             
             float sum_p = 0.0f;
             std::array<float, 26> exp_p;
             
             for(int m=0; m < node.num_legal; ++m) {
                int move = node.legal[m];
                float logit = p_data_base[i*26 + move];
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

             static int _debug_probs = 0;
             if (_debug_probs++ % 1000 == 0) {
                 std::cerr << "[DEBUG NN PROBS] Leaf " << i << " (Softmaxed): ";
                 for(int m=0; m < node.num_legal; ++m) {
                     int move = node.legal[m];
                     std::cerr << move << "=" << node.policy_net[move] << " ";
                 }
                 std::cerr << "\n";
             }
        }
    }
}

void Mcts::run_simulation_finalize(){
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
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid node_idx " << node_idx << " in final_update\n";
        return;
    }
    // 1. Gate Check (Your correct fix)
    if (terminal_set[node_idx] == terminal_epoch){
        terminal_set[node_idx] = 0;
    } else {
        return;
    }

    Node& node = nodes[node_idx];
    if (node.terminal) return;

    // 2. Recurse (Bottom-Up)
    // We use the CORRECT lookup logic here (same as forward_update)
    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];
        int c_idx = node.children[m];
        
        if (c_idx != -1 && terminal_set[c_idx] == terminal_epoch) {
            final_update(c_idx);
        }
    }

    // 3. Pack Active Children (Matching forward_update logic)
    std::array<int, 26> active_children;
    int n_active = 0;

    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];         // <--- THE KEY FIX
        int c_idx = node.children[m];  // <--- THE KEY FIX
        
        if (c_idx != -1) {
            active_children[n_active++] = c_idx;
        }
    }

    if (n_active == 0) return;

    // 4. Calculate Weights
    std::array<float, 26> weights;
    winner_prob(n_active, active_children, weights, param_temperature);
    
    node.reward_target = {0.0f, 0.0f, 0.0f};
    
    for (int i = 0; i < n_active; ++i) {
        int c_idx = active_children[i];
        float w = weights[i];
        const auto& cr = nodes[c_idx].reward_target;
        
        // Parent Loss = Child Win
        node.reward_target[0] += w * cr[2];
        node.reward_target[1] += w * cr[1];
        node.reward_target[2] += w * cr[0];
    }

    // 5. Normalize
    float sum = node.reward_target[0] + node.reward_target[1] + node.reward_target[2];
    if (sum > 1e-9f) {
        float inv = 1.0f / sum;
        node.reward_target[0] *= inv;
        node.reward_target[1] *= inv;
        node.reward_target[2] *= inv;
    }
}
void Mcts::winner_prob(int num_children, const std::array<int, 26>& children, std::array<float, 26>& weights, float temp_scale) {
    weights.fill(0.0f);
    if (num_children == 0) return;
    std::array<float, 26> p0, p1, p2, lt1, lt2;
    for (int i = 0; i < num_children; ++i) {
        int c_idx = children[i];
        if (c_idx < 0 || c_idx >= static_cast<int>(nodes.size())) {
             std::cerr << "ERROR: Invalid child index " << c_idx << " in winner_prob\n";
             // Use default values or skip
             p0[i] = 0.0f; p1[i] = 0.0f; p2[i] = 0.0f;
             lt1[i] = 1e-12f; lt2[i] = 1e-12f;
             continue;
        }
        const auto& q = nodes[c_idx].value_search;
        p0[i] = q[2]; // Win
        p1[i] = q[1]; // Draw
        p2[i] = q[0]; // Loss
        
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
    std::array<float, 26> raw_win;

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
        // Temperature scaling
        if (std::abs(temp_scale - 1.0f) > 1e-6f) {
             float sum_pow = 0.0f;
             if (std::abs(temp_scale) < 1e-9f) {
                 // Zero temperature (argmax)
                 // This logic would need refinement for exact ArgMax, but let's stick to pow for now unless div by zero
                 // If temp is very small, we should just pick the max probability and set it to 1
                 // But let's assume the user doesn't pass exactly 0
                 // If they do, pow might result in INF.
                 float inv_t = 1000.0f; // Soft cap
                 for(int i=0; i<num_children; ++i) {
                     raw_win[i] = std::pow(raw_win[i], inv_t);
                     sum_pow += raw_win[i];
                 }
                 sum_win = sum_pow;
             }
             else {
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
    // static thread_local std::vector<int> leaves_threaded;
    std::vector<int> leaves_threaded;
    leaves_threaded.clear();
    leaves_threaded.reserve(volume);

    forward_recursive(node_idx, volume, leaves_threaded);

    if (!leaves_threaded.empty()) {
        // Validate all leaf indices before inserting
        for (int leaf_idx : leaves_threaded) {
            if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) {
                std::cerr << "ERROR: Invalid leaf_idx " << leaf_idx << " from forward_recursive\n";
                continue;
            }
        }
        // std::lock_guard<std::mutex> lock(mutex);
        leaves.insert(leaves.end(), leaves_threaded.begin(), leaves_threaded.end());
    }
}
void Mcts::process_batch(const std::vector<int>& leaves) {
    if (leaves.empty()) return;
    int batch_size = leaves.size();
    
    // Bounds check
    for (int i = 0; i < batch_size; ++i) {
        int leaf_idx = leaves[i];
        if (leaf_idx < 0 || leaf_idx >= static_cast<int>(nodes.size())) {
            std::cerr << "ERROR: Invalid leaf index " << leaf_idx << " in process_batch\n";
            return;
        }
    }
    
    std::vector<float> input_buffer(batch_size * 50);
    for (int i = 0; i < batch_size; ++i) {
        int leaf_idx = leaves[i];
        nodes[leaf_idx].state.encode_state(&input_buffer[i * 50]);
    }

    auto [p_tens, v_tens, r_tens] = model->inference(input_buffer, batch_size);

    auto p_acc = p_tens.accessor<float, 2>();
    auto v_acc = v_tens.accessor<float, 2>();
    auto r_acc = r_tens.accessor<float, 2>();

    for (int i = 0; i < batch_size; ++i) {
        int idx = leaves[i];
        Node& node = nodes[idx];
        
        if (node.state.is_terminal()) {
            node.terminal = true;
            // terminal_set[idx] = terminal_epoch; 
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
            // NaN Check for Model Output
            if (std::isnan(v_acc[i][0]) || std::isnan(r_acc[i][0])) {
                std::cerr << "WARNING: Model produced NaN values. Zeroing node.\n";
                node.value_net = {0.0f, 0.0f, 0.0f};
                node.reward_net = {0.0f, 0.0f, 0.0f};
            } else {
                std::array<float, 3> value_net = { v_acc[i][0], v_acc[i][1], v_acc[i][2] };
                std::array<float, 3> reward_net = { r_acc[i][0], r_acc[i][1], r_acc[i][2] };
                
                node.value_net = value_net; 
                node.value_search = value_net;
                node.value_target = reward_net; // Bootstrap value target from reward head
                node.reward_net = reward_net;
            }
        } 
        
        float max_logit = -1e9f;
        for(int m=0; m < node.num_legal; ++m) {
            float logit = p_acc[i][node.legal[m]]; 
            if (std::isnan(logit)) {
                std::cerr << "ERROR: NaN logit for node " << idx << " move " << m << "\n";
                logit = -1e9f;
            }
            if (logit > max_logit) max_logit = logit;
        }

        float sum_p = 0.0f;
        std::array<float, 26> exp_p;
        
        for(int m=0; m < node.num_legal; ++m) {
            float logit = p_acc[i][node.legal[m]];
            if (std::isnan(logit)) logit = -1e9f;
            exp_p[m] = std::exp(logit - max_logit); 
            sum_p += exp_p[m];
        }

        if (sum_p < 1e-9f) sum_p = 1e-9f; // Prevent division by zero
        float inv_sum = 1.0f / sum_p;
        for (int m = 0; m < node.num_legal; ++m){
            node.policy_net[node.legal[m]] = exp_p[m]*inv_sum;
            node.policy_search[node.legal[m]] = node.policy_net[node.legal[m]] ;
        }

        node.explored = true;
        
    }
}
void Mcts::mark_traversal_path(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
        return;
    }
    if (node_idx >= static_cast<int>(traversal_set.size())) {
        std::cerr << "ERROR: node_idx " << node_idx << " >= traversal_set.size() " << traversal_set.size() << "\n";
        return;
    }
    if (traversal_set[node_idx] == traversal_epoch) return;
    traversal_set[node_idx] = traversal_epoch;
    
    // Access parents safely with explicit vector access logging
    Node& node = nodes[node_idx];
    int parent_count = node.parents.size();
    
    if (parent_count > 0) {
        // Verify parent_count is reasonable
        if (parent_count > 10000) {
            std::cerr << "ERROR: Node " << node_idx << " has suspiciously large parent count: " << parent_count << "\n";
            return;
        }
        
        for (int i = 0; i < parent_count; ++i) {
            // Safe vector access with bounds check
            if (i >= static_cast<int>(node.parents.size())) {
                std::cerr << "ERROR: parents vector size changed during iteration at node " << node_idx << " (i=" << i << ")\n";
                break;
            }
            int p = node.parents[i];
            if (p >= 0 && p < static_cast<int>(nodes.size())) {
                mark_traversal_path(p);
            }
        }
    }
}
void Mcts::mark_terminal_path(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
        return;
    }
    if (node_idx >= static_cast<int>(terminal_set.size())) {
        std::cerr << "ERROR: node_idx " << node_idx << " >= terminal_set.size() " << terminal_set.size() << "\n";
        return;
    }
    if (terminal_set[node_idx] == terminal_epoch) return;
    terminal_set[node_idx] = terminal_epoch;
    
    // Only recurse if parents vector seems valid
    Node& node = nodes[node_idx];
    int parent_count = node.parents.size();
    
    if (parent_count > 0) {
        // Verify parent_count is reasonable
        if (parent_count > 10000) {
            std::cerr << "ERROR: Node " << node_idx << " has suspiciously large parent count: " << parent_count << "\n";
            return;
        }
        
        for (int i = 0; i < parent_count; ++i) {
            // Safe vector access with bounds check
            if (i >= static_cast<int>(node.parents.size())) {
                std::cerr << "ERROR: parents vector size changed during iteration at node " << node_idx << " (i=" << i << ")\n";
                break;
            }
            int p = node.parents[i];
            if (p >= 0 && p < static_cast<int>(nodes.size())) {
                mark_terminal_path(p);
            }
        }
    }
}
void Mcts::forward_recursive(int node_idx, int volume, std::vector<int>& leaves_threaded){
    if (volume == 0) return;
    
    // Bounds check
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid node_idx " << node_idx << " (nodes.size()=" << nodes.size() << ")\n";
        return;
    }
    
    // Double-check node_count_idx
    if (node_count_idx > static_cast<int>(nodes.size())) {
        std::cerr << "CRITICAL: node_count_idx=" << node_count_idx << " exceeds nodes.size()=" << nodes.size() << "\n";
        return;
    }
    
    Node* node = &nodes[node_idx];
    node->visit_count += volume;
    
    // Sanity check on num_legal
    if (node->num_legal < 0 || node->num_legal > 26) {
        std::cerr << "ERROR: Node " << node_idx << " has invalid num_legal=" << node->num_legal << "\n";
        return;
    }

    if (!node->explored){
        leaves_threaded.push_back(node_idx);
    }else if(node->terminal){
        return;
    }else {
        std::vector<float> probs;
        std::vector<int> prob_index_to_move; 
        
        probs.reserve(node->num_legal + 1);
        prob_index_to_move.reserve(node->num_legal);

        std::vector<pair<float, int>> unexplored;
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

        if (probs.empty()) {
            // All children are terminal or no explored moves available. Stop expansion.
            return;
        }

        static thread_local std::mt19937 rng(std::random_device{}());
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        
        std::vector<int> counts(probs.size(), 0);
        for (int i = 0; i < volume; ++i) {
            int idx = dist(rng);
            if (idx < 0 || idx >= static_cast<int>(counts.size())) {
                std::cerr << "ERROR: dist returned " << idx << " for size " << counts.size() << "\n";
                continue;
            }
            counts[idx]++;
        }

        // Process explored children
        for (int i = 0; i < prob_index_to_move.size(); ++i) {
            if (counts[i] > 0) {
                int move = prob_index_to_move[i];
                int c_idx = node->children[move];
                forward_recursive(c_idx, counts[i], leaves_threaded);
            }
        }

        if (has_bucket && counts[bucket_idx] > 0) {
            std::sort(unexplored.begin(), unexplored.end(), std::greater<pair<float, int>>());
            int particles = counts[bucket_idx];
            for (auto& p : unexplored) {
                if (particles <= 0) break;
                int move = p.second;
                Othello next_state = node->state;
                next_state.apply_action(move);
                uint64_t key = next_state.key();
                int child_idx = -1;
                {
                    // std::lock_guard<std::mutex> lock(mutex);
                    auto it = node_map.find(key);
                    if (it != node_map.end()) {
                        child_idx = it->second;
                        // Only add parent if not already present (avoid duplicates)
                        auto& parents = nodes[child_idx].parents;
                        if (std::find(parents.begin(), parents.end(), node_idx) == parents.end()) {
                            parents.push_back(node_idx);
                        }
                    } else {
                        if (node_count_idx >= nodes.size()) {
                            std::cerr << "CRITICAL: Ran out of node space! Stopping.\n";
                            break; // Just stop instead of crashing
                        }
                        child_idx = node_count_idx++;
                        nodes[child_idx] = Node(); // Reset
                        nodes[child_idx].state = next_state;
                        nodes[child_idx].parents.push_back(node_idx);
                        nodes[child_idx].num_legal = next_state.get_legal_actions(nodes[child_idx].legal);
                        nodes[child_idx].terminal = next_state.is_terminal(); 
                        node_map[key] = child_idx;
                    }
                    node->children[move] = child_idx;
                } // Unlock
                forward_recursive(child_idx, 1, leaves_threaded);
                particles--;
            }
        }
    }
}
void Mcts::forward_update(int node_idx) {
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
        std::cerr << "ERROR: Invalid node_idx " << node_idx << " in forward_update\n";
        return;
    }
    if (traversal_set[node_idx] != traversal_epoch) return;
    traversal_set[node_idx] = 0; // Consume stamp to prevent re-processing

    Node& node = nodes[node_idx];

    if (node.terminal) return;

    std::array<int, 26> active_child_idx;
    std::array<int, 26> active_move_idx;
    int n_active = 0;
    float explored_p = 0.0f;

    for (int i = 0; i < node.num_legal; ++i) {
        int m = node.legal[i];
        int c_idx = node.children[m];

        if (c_idx != -1) {
            if (traversal_set[c_idx] == traversal_epoch) {
                forward_update(c_idx); // Recurse (Bottom-Up Update)
                
                active_child_idx[n_active] = c_idx;
                active_move_idx[n_active] = m;
                n_active++;
                
                explored_p += node.policy_search[m];
            }
        }
    }

    if (n_active == 0) return;

    std::array<float, 26> scaled_probs;
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

        new_v_search[0] += w * cv_search[2]; // Loss += Child Win
        new_v_search[1] += w * cv_search[1]; // Draw += Child Draw
        new_v_search[2] += w * cv_search[0]; // Win  += Child Loss

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
    
    auto fmt = [](int m) -> std::string {
        if (m == 64) return "PASS";
        char col = 'A' + (m % 5);
        char row = '1' + (m / 5);
        return std::string(1, col) + std::string(1, row);
    };

    std::cout << "\n[DEBUG ROOT] Node Idx: " << root_idx << " | Legal Moves: " << root.num_legal << "\n";
    
    // 1. Raw NN Probs
    std::cout << ">>> 1. Raw NN Probs (Policy Net): ";
    for(int i=0; i< std::min(root.num_legal, limit_moves); ++i) {
        int m = root.legal[i];
        std::cout << fmt(m) << "=" << root.policy_net[m] << " ";
    }
    std::cout << "...\n";

    // Reconstruct Search Probs (Need children)
    std::array<int, 26> active_child_idx;
    std::array<int, 26> active_move_idx;
    int n_active = 0;
    float explored_p = 0.0f;

    for (int i = 0; i < root.num_legal; ++i) {
        int m = root.legal[i];
        int c_idx = root.children[m];
        if (c_idx != -1) {
            active_child_idx[n_active] = c_idx;
            active_move_idx[n_active] = m;
            explored_p += root.policy_search[m]; // This might be scaled already, but explored_p is sum
            n_active++;
        }
    }

    if (n_active == 0) {
        std::cout << "No explored children.\n";
        return;
    }

    // 2. Unscaled Search Probs (Temp = 1.0)
    std::array<float, 26> unscaled_weights;
    winner_prob(n_active, active_child_idx, unscaled_weights, 1.0f);
    
    std::cout << ">>> 2. Search Probs (Temp=1.0): ";
    for (int i = 0; i < std::min(n_active, limit_moves); ++i) {
        int m = active_move_idx[i];
        // Note: we just print the weight relative to explored mass
        std::cout << fmt(m) << "=" << unscaled_weights[i] << " ";
    }
    std::cout << "...\n";

    // 3. Scaled Search Probs (Temp = param_temperature)
    std::array<float, 26> scaled_weights;
    winner_prob(n_active, active_child_idx, scaled_weights, param_temperature);

    std::cout << ">>> 3. Search Probs (Temp=" << param_temperature << "): ";
    for (int i = 0; i < std::min(n_active, limit_moves); ++i) {
        int m = active_move_idx[i];
        std::cout << fmt(m) << "=" << scaled_weights[i] << " ";
    }
    std::cout << "...\n\n";
}

