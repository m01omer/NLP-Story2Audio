import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print whether CUDA is available
if cuda_available:
    print(f"CUDA is available. Version: {torch.version.cuda}")
    print(f"Current device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Check your installation.")
