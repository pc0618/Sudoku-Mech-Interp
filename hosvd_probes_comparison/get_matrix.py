import torch
import numpy as np

# Define the model class again to load the state dict
class LinearProbe(torch.nn.Module):
    def __init__(self, input_size=1728, output_size=10):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))

def extract_probe_weights(model_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path)
    
    # Create model instance
    model = LinearProbe(
        input_size=checkpoint['input_size'],
        output_size=checkpoint['output_size']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract weights and bias
    weights = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()
    
    # Save to numpy files
    np.save(f"{model_path.replace('.pt', '_weights.npy')}", weights)
    np.save(f"{model_path.replace('.pt', '_bias.npy')}", bias)
    
    print(f"Weight matrix shape: {weights.shape}")
    print(f"Bias vector shape: {bias.shape}")
    print(f"\nSaved weights to: {model_path.replace('.pt', '_weights.npy')}")
    print(f"Saved bias to: {model_path.replace('.pt', '_bias.npy')}")
    
    return weights, bias

if __name__ == "__main__":
    model_path = "sudoku_probe_model_1.pt"
    weights, bias = extract_probe_weights(model_path)
