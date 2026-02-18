#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <memory>
// #include <mutex>
// #include <atomic>

std::string format_move(int m); // Forward declaration

struct Node ;   
class Model ;   
class Othello ;

class Mcts {
private:
    std::shared_ptr<Model> model;
    std::shared_ptr<Model> model_opp;
    std::vector<int> traversal_set ;
    std::vector<int> terminal_set ;
    std::vector<int> leaves ;
    std::unordered_map<uint64_t , int> node_map ;
    int terminal_epoch ;
    int traversal_epoch ;
    // std::mutex mutex;
    
    void forward(int node_idx, int volume);
    void forward_recursive(int node_idx, int volume, std::vector<int>& local_leaves);
    void process_batch(const std::vector<int>& leaves);
    void forward_update(int node_idx);
    void mark_traversal_path(int node_idx);
    void mark_terminal_path(int node_idx);
    void winner_prob(int num_children, const std::array<int, 26>& children, std::array<float, 26>& weights, float temp_scale);
public:
    Mcts();
    ~Mcts();
    void print_debug_root(int limit_moves = 5);
    std::vector<Node> nodes ;
    int root_idx ;
    // std::atomic<int> node_count_idx ;
    int node_count_idx ;
    float param_temperature = 1.0f;
    bool is_game_over();
    void set_temperature(float temp) { param_temperature = temp; }
    void load_model(const std::string& path);
    void load_model(const std::string& path, const std::string& path_opp);
    void set_model(std::shared_ptr<Model> m) { model = m; }
    void initialize(const Othello& game);
    int get_move(bool deterministic);
    void advance_root(int move_idx) ;
    void run_simulation_batch(int volume);
    
    // External batching support
    void run_simulation_prepare(int volume);
    int get_leaf_count() const { return leaves.size(); }
    void extract_leaf_states(float* buffer); 
    void update_leaves_from_tensor(const float* p_data_base, const float* v_data_base, const float* r_data_base, int batch_size);
    void run_simulation_finalize();

    void final_update(int node_idx);
};
