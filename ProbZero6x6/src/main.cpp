#include "selfplay.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

// --- BINARY SERIALIZER ---
// Format: [N_SAMPLES (int32)] [SAMPLE_1 (74 floats)] ...
// Per Sample Layout (4x4):
//   - Board:         32 floats (2 planes of 4x4)
//   - Policy:        37 floats (16 squares + pass)
//   - Mask:          37 floats
//   - Value Target:   3 floats
//   - Reward Target:  3 floats
//   - Reward Valid:   1 float
//   - Weight:         1 float
//   Total: 32+37+37+3+3+1+1 = 74 floats
void save_dataset_binary(const std::vector<DataFormat>& dataset, const std::string& filename) {
    if (dataset.empty()) {
        std::cerr << "Warning: Dataset is empty. Nothing to save.\n";
        return;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    int32_t n_samples = static_cast<int32_t>(dataset.size());
    out.write(reinterpret_cast<const char*>(&n_samples), sizeof(int32_t));

    for (const auto& sample : dataset) {
        out.write(reinterpret_cast<const char*>(sample.board.data()),       72 * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.policy.data()),      37 * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.policy_mask.data()), 37 * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.target_q.data()),     3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(sample.target_r.data()),     3 * sizeof(float));
        out.write(reinterpret_cast<const char*>(&sample.reward_valid),           sizeof(float));
        out.write(reinterpret_cast<const char*>(&sample.weight),                 sizeof(float));
    }

    out.close();
    std::cout << "Successfully saved " << n_samples << " samples to " << filename
              << " (" << (n_samples * 154 * 4 / 1024 / 1024) << " MB)\n";
}

int main(int argc, char* argv[]) {
    std::string model_path  = "model_script.pt";
    std::string output_file = "dataset.bin";
    int batch_size   = 8;
    int num_batches  = 60;
    int num_games    = 20;
    int cycle        = 0;

    if (argc > 1) model_path  = argv[1];
    if (argc > 2) output_file = argv[2];
    if (argc > 3) num_batches = std::stoi(argv[3]);
    if (argc > 4) num_games   = std::stoi(argv[4]);

    const char* env_iter = std::getenv("ITERATION");
    if (env_iter) cycle = std::stoi(env_iter);

    std::cout << "=== Othello 4x4 AlphaZero Tree Harvester ===\n";
    std::cout << "Model: "           << model_path  << "\n";
    std::cout << "Target Batches: "  << num_batches << "\n";
    std::cout << "Particles/Batch: " << batch_size  << "\n";
    std::cout << "Cycle: "           << cycle       << "\n";
    std::cout << "Parallel Games: "  << num_games   << "\n";

    try {
        Selfplay engine(model_path, num_games, cycle);

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<DataFormat> dataset;
        engine.generate_tree_data(batch_size, num_batches, dataset);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "----------------------------------------\n";
        std::cout << "Time Elapsed: "        << elapsed.count()                     << "s\n";
        std::cout << "Positions Generated: " << dataset.size()                      << "\n";
        std::cout << "Speed: "               << (dataset.size() / elapsed.count())  << " positions/sec\n";

        save_dataset_binary(dataset, output_file);

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
