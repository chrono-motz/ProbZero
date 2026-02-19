#pragma once
#include "game.hpp"
#include <array>
#include <vector>

struct Node {
    bool explored;
    bool terminal;
    Othello state;
    
    // Initialize ALL arrays to 0.0f to prevent garbage/NaNs
    std::array<float,3> value_net{};
    std::array<float,3> value_search{};
    std::array<float,3> value_target{};
    std::array<float,3> reward_net{};
    std::array<float,3> reward_target{};
    std::array<float,26> policy_net{};
    std::array<float,26> policy_search{};
    
    std::array<int,26> legal;
    std::array<int,26> children;
    int num_legal;
    std::vector<int> parents;
    float visit_count;

    Node() {
        // Explicitly fill arrays if {} above isn't supported by compiler, 
        // but {} is standard C++11 for zero-init. 
        // We'll use fill() to be absolutely safe.
        value_net.fill(0.0f);
        value_search.fill(0.0f);
        value_target.fill(0.0f);
        reward_net.fill(0.0f);
        reward_target.fill(0.0f);
        policy_net.fill(0.0f);
        policy_search.fill(0.0f);
        
        children.fill(-1);
        explored = false;
        terminal = false;
        visit_count = 0.0f;
    }
};