#pragma once
#include "mcts.hpp"
#include "game.hpp"
#include "node.hpp"
#include "model.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>

struct DataFormat {
    std::vector<float> board;      
    std::vector<float> policy;     
    std::vector<float> policy_mask; 
    std::vector<float> target_q;   
    std::vector<float> target_r;   
    float reward_valid;            
    float weight;
};

struct AccumulatedData {
    std::vector<float> board;    
    std::vector<float> policy_accum; 
    std::vector<float> policy_mask; 
    std::vector<float> q_accum;   
    std::vector<float> r_accum;   
    float valid_sum;            
    float weight_sum;
    
    AccumulatedData() : weight_sum(0.0f), valid_sum(0.0f) {}
};

class Selfplay {
private:
    std::shared_ptr<Model> model;
    std::vector<std::unique_ptr<Mcts>> games;
    std::unordered_map<uint64_t, AccumulatedData> global_map;

    void extract_and_unify(Mcts* mcts) {
        for(int i=0; i < mcts->node_count_idx; ++i) {
            const Node& node = mcts->nodes[i];
            if (!node.explored) continue;
            
            bool has_children = false;
            for(int c : node.children) {
                if(c != -1) { has_children = true; break; }
            }
            if(!has_children) continue;

            uint64_t key = node.state.key(); 
            AccumulatedData& entry = global_map[key];
            
            if (entry.weight_sum == 0.0f) {
                 // First time seeing this state
                 entry.board.resize(50);
                 Othello temp = node.state;
                 temp.encode_state(entry.board.data());
                 
                 entry.policy_accum.resize(26, 0.0f);
                 entry.policy_mask.resize(26, 0.0f);
                 entry.q_accum.resize(3, 0.0f);
                 entry.r_accum.resize(3, 0.0f);
            }
            
            // FIX: Update mask for EVERY visit, not just the first one.
            // If Game 2 explores a move Game 1 didn't, we need to unmask it.
            for(int m=0; m<26; ++m) {
                if (node.children[m] != -1) entry.policy_mask[m] = 1.0f;
            }
            
            float w = node.visit_count;
            if (w < 1e-6f) w = 1.0f;
            
            // for(int m=0; m<26; ++m) entry.policy_accum[m] += node.policy_search[m] * w;
            for(int m=0; m<26; ++m) {
                // ONLY accumulate policy if the move actually has a child (was visited).
                // Previously, this line blindly added 'node.policy_search[m]', which contained
                // raw neural net priors for unvisited moves!
                if (node.children[m] != -1) { 
                    entry.policy_accum[m] += node.policy_search[m] * w;
                }
            }
            
            for(int k=0; k<3; ++k) {
                entry.q_accum[k] += node.value_target[k] * w;
                entry.r_accum[k] += node.reward_target[k] * w;
            }
            
            float r_sum = node.reward_target[0] + node.reward_target[1] + node.reward_target[2];
            float is_valid = (r_sum > 1e-6f) ? 1.0f : 0.0f;
            entry.valid_sum += is_valid * w;
            
            entry.weight_sum += w;
        }
    }

public:
    Selfplay(const std::string& model_path, int num_games = 1) {
        model = std::make_shared<Model>();
        model->load_model(model_path);
        
        games.resize(num_games);
        for(int i=0; i<num_games; ++i) {
            games[i] = std::make_unique<Mcts>();
            games[i]->set_model(model);
            Othello g; 
            games[i]->initialize(g);
        }
    }
    
    void generate_tree_data(int flow_volume, int num_batches_per_move, std::vector<DataFormat>& dataset) {
         // Reset games handled by recreate or reinit? 
         // Helper to reinit same Mcts object:
         for(auto& g : games) {
             Othello o; 
             g->initialize(o);
         }
         
         global_map.clear();
         
         bool any_active = true;
         int iteration = 0;
         
         std::cout << "Starting Parallel Selfplay (" << games.size() << " games)...\n";

         while(any_active) {
             iteration++;
             if (iteration % 10 == 0) std::cout << "Move " << iteration << "...\n";
             
             // 1. Expansion Phrase (Batched)
             for(int b=0; b < num_batches_per_move; ++b) {
                 int total_leaves = 0;
                 for(auto& mcts : games) {
                     if (!mcts->is_game_over()) {
                         mcts->run_simulation_prepare(flow_volume);
                         total_leaves += mcts->get_leaf_count();
                     }
                 }
                 
                 if (total_leaves == 0) continue;
                 
                 std::vector<float> input_buffer(total_leaves * 50);
                 float* ptr = input_buffer.data();
                 
                 for(auto& mcts : games) {
                     if (!mcts->is_game_over()) {
                         mcts->extract_leaf_states(ptr);
                         ptr += mcts->get_leaf_count() * 50;
                     }
                 }
                 
                 auto result_tuple = model->inference(input_buffer, total_leaves);
                 // unpacking tuple: (P, V, R)
                 auto& p_tens = std::get<0>(result_tuple);
                 auto& v_tens = std::get<1>(result_tuple);
                 auto& r_tens = std::get<2>(result_tuple);

                 // Data pointers (assuming CPU tensors per fixed model.hpp)
                 // Need to ensure continuous. toTensor() moves it, they should be contiguous.
                 // safe to get data_ptr
                 const float* p_ptr = p_tens.data_ptr<float>();
                 const float* v_ptr = v_tens.data_ptr<float>();
                 const float* r_ptr = r_tens.data_ptr<float>();
                 
                 int offset = 0;
                 for(auto& mcts : games) {
                     if (!mcts->is_game_over()) {
                         int count = mcts->get_leaf_count();
                         if (count > 0) {
                             mcts->update_leaves_from_tensor(
                                 p_ptr + offset * 26, 
                                 v_ptr + offset * 3, 
                                 r_ptr + offset * 3, 
                                 count
                             );
                             mcts->run_simulation_finalize();
                             offset += count;
                         }
                     }
                 }
             }
             
             // 2. Advance Games
             any_active = false;
             for(auto& mcts : games) {
                 if (!mcts->is_game_over()) {
                     int m = mcts->get_move(false); // stochastic=false -> deterministic? 
                                                    // No, for self-play training, we usually want exploration.
                                                    // This code previously used false (deterministic=false -> stochastic).
                                                    // So we keep false.
                     if (m >= 0) {
                         mcts->advance_root(m);
                         any_active = true;
                     } 
                 }
             }
         }
         
         std::cout << "All games complete. Aggregating data...\n";
         
         // Collect Data
         for(auto& mcts : games) {
              mcts->final_update(0);
              extract_and_unify(mcts.get());
         }

         // Convert map to dataset
         dataset.reserve(dataset.size() + global_map.size());
         
         for(auto& kv : global_map) {
             const AccumulatedData& acc = kv.second;
             DataFormat d;
             d.board = acc.board;
             d.policy_mask = acc.policy_mask;
             d.weight = acc.weight_sum;
             
             d.policy = acc.policy_accum;
             d.target_q = acc.q_accum;
             d.target_r = acc.r_accum;
             
             // Normalize
             if (d.weight > 1e-9f) {
                 float inv_w = 1.0f / d.weight;
                 for(auto& x : d.policy) x *= inv_w;
                 for(auto& x : d.target_q) x *= inv_w;
                 for(auto& x : d.target_r) x *= inv_w;
                 d.reward_valid = acc.valid_sum * inv_w;
             } else {
                 d.reward_valid = 0.0f;
             }
             
             dataset.push_back(d);
         }
         std::cout << "Extracted " << dataset.size() << " unique positions.\n";
    }
};