import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Using device:", device)

# Create large tensors on GPU
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()

start = time.time()

c = torch.matmul(a, b)

torch.cuda.synchronize()

end = time.time()

print(f"Matrix multiplication took: {end - start:.3f} seconds")
print("GPU memory allocated:", round(
    torch.cuda.memory_allocated() / 1024**2, 2), "MB")
