import torch
import torch.nn as nn
import torch.nn.functional as F
from train import OthelloNet, DEVICE

# Helper to save as C++ compatible TorchScript
def save_traced(model, filename):
    model.cpu().eval()
    # Dummy input must match your C++ input shape exactly: (Batch=1, Channels=2, H=8, W=8)
    example = torch.rand(1, 2, 5, 5)
    
    class ExportWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            p, v, r = self.m(x)
            # Apply Softmax here so C++ gets probabilities directly
            return p, F.softmax(v, dim=1), F.softmax(r, dim=1)
            
    traced = torch.jit.trace(ExportWrapper(model), example)
    traced.save(filename)
    print(f"Converted -> {filename}")

def convert(weights_path, output_path):
    print(f"Loading {weights_path}...")
    model = OthelloNet().to(DEVICE)
    try:
        # Load the Python weights
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    save_traced(model, output_path)

if __name__ == "__main__":
    # Convert the specific files you wanted to use
    convert("weights_cycle_10.pth",  "model_cycle_10.pt")
    convert("weights_cycle_100.pth", "model_cycle_100.pt")