#pragma once
#include <cstdint>
#include <array>
#include <algorithm>

// 4x4 Othello (Mini-Othello) for rapid training
// Board layout: 16 squares (0-15), row-major
// Starting position (standard 4x4 Othello center):
//   . . . .
//   . O X .
//   . X O .
//   . . . .
// Black(X): sq 6,9  White(O): sq 5,10

class Othello {
private:
    uint32_t bitboards[2]; // [0]=Black, [1]=White (only need 16 bits)
    int turn;              // 0 or 1

    // For a 4-wide board, stride=4.
    // Column masks to prevent wrap-around:
    //   "not left column"  (col 0 = bits 0,4,8,12)  -> mask = 0b1110_1110_1110_1110 = 0xEEEE
    //   "not right column" (col 3 = bits 3,7,11,15) -> mask = 0b0111_0111_0111_0111 = 0x7777
    inline uint32_t shift(uint32_t b, int dir) const {
        switch(dir) {
            case  4: return  b << 4;                         // Down
            case -4: return  b >> 4;                         // Up
            case  1: return (b << 1) & 0xEEEEU;             // Right  (mask left col)
            case -1: return (b >> 1) & 0x7777U;             // Left   (mask right col)
            case  5: return (b << 5) & 0xEEEEU;             // Down-right
            case -5: return (b >> 5) & 0x7777U;             // Up-left
            case  3: return (b << 3) & 0x7777U;             // Down-left
            case -3: return (b >> 3) & 0xEEEEU;             // Up-right
            default: return 0;
        }
    }

    inline uint32_t get_legal_mask() const {
        uint32_t self  = bitboards[turn];
        uint32_t opp   = bitboards[1 - turn];
        uint32_t empty = ~(self | opp) & 0xFFFFU; // Mask to 16 bits
        uint32_t legal = 0;
        int dirs[8] = {4, -4, 1, -1, 5, -5, 3, -3};

        for (int dir : dirs) {
            uint32_t x = shift(self, dir) & opp;
            x |= shift(x, dir) & opp;
            x |= shift(x, dir) & opp;
            legal |= shift(x, dir) & empty;
        }
        return legal;
    }

public:
    Othello() : turn(0) {
        // Standard 4x4 starting position:
        //   row 1, col 1 = sq 5  (White/O)
        //   row 1, col 2 = sq 6  (Black/X)
        //   row 2, col 1 = sq 9  (Black/X)
        //   row 2, col 2 = sq 10 (White/O)
        bitboards[0] = (1U << 6) | (1U << 9);   // Black
        bitboards[1] = (1U << 5) | (1U << 10);  // White
    }

    // Encodes as: [My Pieces (16), Opponent Pieces (16)]
    void encode_state(float* buffer) const {
        for (int i = 0; i < 16; ++i) {
            buffer[i]      = (bitboards[turn]     >> i) & 1;
            buffer[16 + i] = (bitboards[1 - turn] >> i) & 1;
        }
    }

    inline int current_player() const { return turn; }

    inline void apply_action(int sq) {
        if (sq == 16) { turn = 1 - turn; return; } // Pass

        uint32_t self = bitboards[turn];
        uint32_t opp  = bitboards[1 - turn];
        uint32_t flips = 0;

        int r = sq / 4, c = sq % 4;
        int dirs[8][2] = {{-1,0}, {-1,1}, {0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}};

        for (auto& d : dirs) {
            uint32_t dir_flips = 0;
            int tr = r + d[0], tc = c + d[1];
            while (tr >= 0 && tr < 4 && tc >= 0 && tc < 4) {
                int tsq = tr * 4 + tc;
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

        bitboards[turn]     |= ((1U << sq) | flips);
        bitboards[1 - turn] &= ~flips;
        turn = 1 - turn;
    }

    inline int get_legal_actions(std::array<int, 17>& moves) const {
        uint32_t mask = get_legal_mask();
        int count = 0;
        while (mask) {
            int sq = __builtin_ctz(mask);
            moves[count++] = sq;
            mask &= mask - 1;
        }
        // Handle Pass (16) if no moves but game not over
        if (count == 0 && (bitboards[0] | bitboards[1]) != 0xFFFFU) {
            moves[count++] = 16;
        }
        return count;
    }

    inline bool check_win(int p) const {
        if (!is_terminal()) return false;
        return __builtin_popcount(bitboards[p]) > __builtin_popcount(bitboards[1 - p]);
    }

    inline bool is_terminal() const {
        if ((bitboards[0] | bitboards[1]) == 0xFFFFU) return true; // Board full
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
