#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>

class Othello {
private:
    uint64_t bitboards[2]; // [0]=Black, [1]=White (Uses up to 64 bits)
    int turn;              // 0 or 1
    int board_size;        // 4, 6, 8
    uint64_t board_mask;   // Mask for valid board bits
    uint64_t left_col_mask;  // Mask for all except left column
    uint64_t right_col_mask; // Mask for all except right column

    void init_masks() {
        board_mask = (board_size * board_size == 64) ? ~0ULL : ((1ULL << (board_size * board_size)) - 1);
        left_col_mask = 0;
        right_col_mask = 0;
        for (int r = 0; r < board_size; ++r) {
            for (int c = 0; c < board_size; ++c) {
                if (c > 0) left_col_mask |= (1ULL << (r * board_size + c));
                if (c < board_size - 1) right_col_mask |= (1ULL << (r * board_size + c));
            }
        }
    }

    inline uint64_t shift(uint64_t b, int dir) const {
        switch(dir) {
            case 0: return  b << board_size;                                  // Down
            case 1: return  b >> board_size;                                  // Up
            case 2: return (b << 1) & left_col_mask;                          // Right (mask left col)
            case 3: return (b >> 1) & right_col_mask;                         // Left  (mask right col)
            case 4: return (b << (board_size + 1)) & left_col_mask;           // Down-right
            case 5: return (b >> (board_size + 1)) & right_col_mask;          // Up-left
            case 6: return (b << (board_size - 1)) & right_col_mask;          // Down-left
            case 7: return (b >> (board_size - 1)) & left_col_mask;           // Up-right
            default: return 0;
        }
    }

    inline uint64_t get_legal_mask() const {
        uint64_t self  = bitboards[turn];
        uint64_t opp   = bitboards[1 - turn];
        uint64_t empty = ~(self | opp) & board_mask;
        uint64_t legal = 0;

        for (int dir = 0; dir < 8; ++dir) {
            uint64_t x = shift(self, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp; 
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp; // Max 7 steps possible on 8x8
            legal |= shift(x, dir) & empty;
        }
        return legal;
    }

public:
    Othello(int size) : turn(0), board_size(size) {
        init_masks();
        bitboards[0] = 0;
        bitboards[1] = 0;
        
        int r1 = board_size / 2 - 1;
        int r2 = board_size / 2;
        int c1 = board_size / 2 - 1;
        int c2 = board_size / 2;
        
        // Standard center cross alignment:
        bitboards[0] = (1ULL << (r1 * board_size + c2)) | (1ULL << (r2 * board_size + c1)); // Black
        bitboards[1] = (1ULL << (r1 * board_size + c1)) | (1ULL << (r2 * board_size + c2)); // White
    }

    void encode_state(float* buffer) const {
        int sqs = board_size * board_size;
        for (int i = 0; i < sqs; ++i) {
            buffer[i]       = (bitboards[turn]     >> i) & 1;
            buffer[sqs + i] = (bitboards[1 - turn] >> i) & 1;
        }
    }

    inline int current_player() const { return turn; }
    inline int get_board_size() const { return board_size; }
    inline int get_num_actions() const { return board_size * board_size + 1; }

    inline void apply_action(int sq) {
        int pass_sq = board_size * board_size;
        if (sq == pass_sq) { turn = 1 - turn; return; } // Pass

        uint64_t self = bitboards[turn];
        uint64_t opp  = bitboards[1 - turn];
        uint64_t flips = 0;

        int r = sq / board_size, c = sq % board_size;
        int dirs[8][2] = {{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}};

        for (auto& d : dirs) {
            uint64_t dir_flips = 0;
            int tr = r + d[0], tc = c + d[1];
            while (tr >= 0 && tr < board_size && tc >= 0 && tc < board_size) {
                int tsq = tr * board_size + tc;
                if ((opp >> tsq) & 1) {
                    dir_flips |= (1ULL << tsq);
                } else if ((self >> tsq) & 1) {
                    flips |= dir_flips;
                    break;
                } else {
                    break;
                }
                tr += d[0]; tc += d[1];
            }
        }

        bitboards[turn]     |= ((1ULL << sq) | flips);
        bitboards[1 - turn] &= ~flips;
        turn = 1 - turn;
    }

    inline int get_legal_actions(std::vector<int>& moves) const {
        moves.clear();
        uint64_t mask = get_legal_mask();
        while (mask) {
            int sq = __builtin_ctzll(mask);
            moves.push_back(sq);
            mask &= mask - 1;
        }
        if (moves.empty() && (bitboards[0] | bitboards[1]) != board_mask) {
            moves.push_back(board_size * board_size); // Pass action index
        }
        return moves.size();
    }

    inline bool check_win(int p) const {
        if (!is_terminal()) return false;
        return __builtin_popcountll(bitboards[p]) > __builtin_popcountll(bitboards[1 - p]);
    }

    inline bool is_terminal() const {
        if ((bitboards[0] | bitboards[1]) == board_mask) return true;
        if (get_legal_mask() != 0) return false;

        Othello temp = *this;
        temp.turn = 1 - temp.turn;
        return temp.get_legal_mask() == 0;
    }

    inline uint64_t key() const {
        constexpr uint64_t k1 = 0x9e3779b97f4a7c15ULL;
        constexpr uint64_t k2 = 0xbf58476d1ce4e5b9ULL;
        uint64_t h = bitboards[0] * k1 ^ bitboards[1] * k2 ^ (uint64_t(turn) << 63);
        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL; h ^= (h >> 33);
        return h;
    }

    void restore_state(uint64_t b0, uint64_t b1, int t) {
        bitboards[0] = b0;
        bitboards[1] = b1;
        turn = t;
    }
    
    uint64_t get_bitboard(int p) const {
        return bitboards[p];
    }
};
