#pragma once
#include "game.hpp"
#include <array>
#include <vector>

struct Node {
    bool explored;
    bool terminal;
    Othello state;
    
    // Initialize arrays to 0.0f
    std::array<float,3> value_net{};
    std::array<float,3> value_search{};
    std::array<float,3> value_target{};
    std::array<float,3> reward_net{};
    std::array<float,3> reward_target{};
    
    std::vector<float> policy_net;
    std::vector<float> policy_search;
    
    std::vector<int> legal;
    std::vector<int> children;
    int num_legal;
    std::vector<int> parents;
    float visit_count;

    Node(int board_size) : state(board_size) {
        value_net.fill(0.0f);
        value_search.fill(0.0f);
        value_target.fill(0.0f);
        reward_net.fill(0.0f);
        reward_target.fill(0.0f);
        
        int num_actions = board_size * board_size + 1;
        policy_net.resize(num_actions, 0.0f);
        policy_search.resize(num_actions, 0.0f);
        
        children.resize(num_actions, -1);
        
        explored = false;
        terminal = false;
        visit_count = 0.0f;
    }
};
