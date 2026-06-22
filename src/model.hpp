#pragma once

#include <torch/script.h> 
#include <vector>
#include <string>

class Model {
private:
    mutable torch::jit::script::Module module;
    torch::Device device; 

public:
    Model() : device(torch::kCUDA) {}
    
    void load_model(const std::string& path) {
        module = torch::jit::load(path, device);
        module.eval();
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
    inference(const std::vector<float>& input_data, int batch_size, int board_size) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob((void*)input_data.data(), {batch_size, 2, board_size, board_size}, options).to(device);
        
        torch::NoGradGuard no_grad;
        auto outputs = module.forward({input}).toTuple();
        
        return {
            outputs->elements()[0].toTensor().to(torch::kCPU),
            outputs->elements()[1].toTensor().to(torch::kCPU),
            outputs->elements()[2].toTensor().to(torch::kCPU)
        };
    }
};
