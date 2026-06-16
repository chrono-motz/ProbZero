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
    std::array<float,17> policy_net{};
    std::array<float,17> policy_search{};
    
    std::array<int,17> legal;
    std::array<int,17> children;
    int num_legal;
    std::vector<int> parents;
    float visit_count;

    Node() {
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
