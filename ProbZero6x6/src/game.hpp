#pragma once
#include <cstdint>
#include <array>
#include <algorithm>
#include <iostream>

class Othello {
private:
    uint64_t bitboards[2]; // [0]=Black, [1]=White (Uses 36 bits)
    int turn;              // 0 or 1

    inline uint64_t shift(uint64_t b, int dir) const {
        switch(dir) {
            case  6: return  b << 6;                                 // Down
            case -6: return  b >> 6;                                 // Up
            case  1: return (b << 1) & 0xFBEFBEFBEULL;               // Right (mask left col)
            case -1: return (b >> 1) & 0x7DF7DF7DFULL;               // Left  (mask right col)
            case  7: return (b << 7) & 0xFBEFBEFBEULL;               // Down-right
            case -7: return (b >> 7) & 0x7DF7DF7DFULL;               // Up-left
            case  5: return (b << 5) & 0x7DF7DF7DFULL;               // Down-left
            case -5: return (b >> 5) & 0xFBEFBEFBEULL;               // Up-right
            default: return 0;
        }
    }

    inline uint64_t get_legal_mask() const {
        uint64_t self  = bitboards[turn];
        uint64_t opp   = bitboards[1 - turn];
        uint64_t empty = ~(self | opp) & 0xFFFFFFFFFULL; // Mask to 36 bits
        uint64_t legal = 0;
        int dirs[8] = {6, -6, 1, -1, 7, -7, 5, -5};

        for (int dir : dirs) {
            uint64_t x = shift(self, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp; // Max 5 steps possible on 6x6
            legal |= shift(x, dir) & empty;
        }
        return legal;
    }

public:
    Othello() : turn(0) {
        // Standard 6x6 center cross alignment:
        // Row 2, Col 2 = sq 14 (White) | Row 2, Col 3 = sq 15 (Black)
        // Row 3, Col 2 = sq 20 (Black) | Row 3, Col 3 = sq 21 (White)
        bitboards[0] = (1ULL << 15) | (1ULL << 20);  // Black
        bitboards[1] = (1ULL << 14) | (1ULL << 21);  // White
    }

    void encode_state(float* buffer) const {
        for (int i = 0; i < 36; ++i) {
            buffer[i]      = (bitboards[turn]     >> i) & 1;
            buffer[36 + i] = (bitboards[1 - turn] >> i) & 1;
        }
    }

    inline int current_player() const { return turn; }

    inline void apply_action(int sq) {
        if (sq == 36) { turn = 1 - turn; return; } // Pass

        uint64_t self = bitboards[turn];
        uint64_t opp  = bitboards[1 - turn];
        uint64_t flips = 0;

        int r = sq / 6, c = sq % 6;
        int dirs[8][2] = {{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}};

        for (auto& d : dirs) {
            uint64_t dir_flips = 0;
            int tr = r + d[0], tc = c + d[1];
            while (tr >= 0 && tr < 6 && tc >= 0 && tc < 6) {
                int tsq = tr * 6 + tc;
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

    inline int get_legal_actions(std::array<int, 37>& moves) const {
        uint64_t mask = get_legal_mask();
        int count = 0;
        while (mask) {
            int sq = __builtin_ctzll(mask);
            moves[count++] = sq;
            mask &= mask - 1;
        }
        if (count == 0 && (bitboards[0] | bitboards[1]) != 0xFFFFFFFFFULL) {
            moves[count++] = 36; // Pass action index
        }
        return count;
    }

    inline bool check_win(int p) const {
        if (!is_terminal()) return false;
        return __builtin_popcountll(bitboards[p]) > __builtin_popcountll(bitboards[1 - p]);
    }

    inline bool is_terminal() const {
        if ((bitboards[0] | bitboards[1]) == 0xFFFFFFFFFULL) return true;
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
};