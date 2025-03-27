import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    """A simple 3-layer neural network."""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5)
        )
    
    def forward(self, x):
        return self.layers(x)

def test_gpus():
    """Test accessibility of all 8 GPUs (0-7) and report which ones are available."""
    # First check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return
    
    # Get total number of GPUs
    total_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected by PyTorch: {total_gpus}")
    
    # Create simple model
    model = SimpleNN()
    
    # Try to access each GPU
    accessible_gpus = []
    for gpu_id in range(8):
        try:
            # Try to move model to this GPU
            device = torch.device(f"cuda:{gpu_id}")
            model_on_gpu = model.to(device)
            
            # Create a small input tensor and perform a forward pass
            x = torch.randn(1, 10).to(device)
            output = model_on_gpu(x)
            
            # If everything worked, this GPU is accessible
            print(f"GPU {gpu_id} is accessible ✓")
            accessible_gpus.append(gpu_id)
        except Exception as e:
            print(f"GPU {gpu_id} is not accessible ✗ - Error: {str(e)}")
    
    # Print summary
    print("\nSummary:")
    print(f"Accessible GPUs: {accessible_gpus}")
    print(f"Total accessible GPUs: {len(accessible_gpus)}")
    
    return accessible_gpus

if __name__ == "__main__":
    test_gpus() 