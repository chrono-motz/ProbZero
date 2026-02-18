#pragma once
#include <cstdint>
#include <array>
#include <algorithm>

// 5x5 Othello (Mini-Othello) for rapid training
// Board layout: 25 squares (0-24)
// Starting position:
//   . . . . .
//   . . . . .
//   . X O . .
//   . O X . .
//   . . . . .

class Othello {
private:
    uint32_t bitboards[2]; // [0]=Black, [1]=White (only need 25 bits)
    int turn;              // 0 or 1

    inline uint32_t shift(uint32_t b, int dir) const {
        switch(dir) {
            case 5: return b << 5;                          // Down
            case -5: return b >> 5;                         // Up
            case 1: return (b << 1) & 0xEFDEFBDEUL;        // Right (mask left column)
            case -1: return (b >> 1) & 0x7BEF7BEFUL;       // Left (mask right column)
            case 6: return (b << 6) & 0xEFDEFBDEUL;        // Down-right
            case -6: return (b >> 6) & 0x7BEF7BEFUL;       // Up-left
            case 4: return (b << 4) & 0x7BEF7BEFUL;        // Down-left
            case -4: return (b >> 4) & 0xEFDEFBDEUL;       // Up-right
            default: return 0;
        }
    }

    inline uint32_t get_legal_mask() const {
        uint32_t self = bitboards[turn];
        uint32_t opp = bitboards[1 - turn];
        uint32_t empty = ~(self | opp) & 0x1FFFFFFUL; // Mask to 25 bits
        uint32_t legal = 0;
        int dirs[8] = {5, -5, 1, -1, 6, -6, 4, -4};
        
        for (int dir : dirs) {
            uint32_t x = shift(self, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            legal |= shift(x, dir) & empty;
        }
        return legal;
    }

public:
    Othello() : turn(0) {
        // Starting position (center 2x2)
        // Squares 7, 12 (Black), 11, 8 (White)
        bitboards[0] = (1U << 7) | (1U << 12);  // Black: C3, D2
        bitboards[1] = (1U << 11) | (1U << 8);  // White: D3, C2
    }

    // Encodes as: [My Pieces, Opponent Pieces]
    void encode_state(float* buffer) const {
        for (int i = 0; i < 25; ++i) {
            buffer[i] = (bitboards[turn] >> i) & 1;        
            buffer[25 + i] = (bitboards[1-turn] >> i) & 1; 
        }
    }
    
    inline int current_player() const { return turn; }

    inline void apply_action(int sq) {
        if (sq == 25) { turn = 1 - turn; return; } // Pass

        uint32_t self = bitboards[turn];
        uint32_t opp = bitboards[1 - turn];
        uint32_t flips = 0;
        uint32_t move_bit = 1U << sq;

        int r = sq / 5, c = sq % 5;
        int dirs[8][2] = {{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}};

        for(auto& d : dirs) {
            uint32_t dir_flips = 0;
            int tr = r + d[0], tc = c + d[1];
            while (tr >= 0 && tr < 5 && tc >= 0 && tc < 5) {
                int tsq = tr * 5 + tc;
                if ((opp >> tsq) & 1) {
                    dir_flips |= (1U << tsq);
                } else if ((self >> tsq) & 1) {
                    flips |= dir_flips;
                    break;
                } else {
                    break;
                }
                tr += d[0]; tc += d[1];
            }
        }
        
        bitboards[turn] |= (move_bit | flips);
        bitboards[1 - turn] &= ~flips;
        turn = 1 - turn;
    }

    inline int get_legal_actions(std::array<int, 26>& moves) const {
        uint32_t mask = get_legal_mask();
        int count = 0;
        while (mask) {
            int sq = __builtin_ctz(mask);
            moves[count++] = sq;
            mask &= mask - 1;
        }
        // Handle Pass (25) if no moves but game not over
        if (count == 0 && (bitboards[0] | bitboards[1]) != 0x1FFFFFFUL) {
             moves[count++] = 25; 
        }
        return count;
    }

    inline bool check_win(int p) const {
        if (!is_terminal()) return false; 
        return __builtin_popcount(bitboards[p]) > __builtin_popcount(bitboards[1-p]);
    }

    inline bool is_terminal() const {
        if ((bitboards[0] | bitboards[1]) == 0x1FFFFFFUL) return true; // Board full
        if (get_legal_mask() != 0) return false;
        
        // Check opponent moves
        Othello temp = *this;
        temp.turn = 1 - temp.turn;
        return temp.get_legal_mask() == 0;
    }

    // Collision-resistant hash
    inline uint64_t key() const {
        constexpr uint64_t k1 = 0x9e3779b97f4a7c15ULL;
        constexpr uint64_t k2 = 0xbf58476d1ce4e5b9ULL;
        
        uint64_t h = uint64_t(bitboards[0]) * k1;
        h ^= uint64_t(bitboards[1]) * k2;
        h ^= (uint64_t(turn) << 63);
        
        h ^= (h >> 33);
        h *= 0xff51afd7ed558ccdULL;
        h ^= (h >> 33);
        
        return h;
    }
};