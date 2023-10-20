import torch

batch_size = 128
num_heads = 8
N = 256
hidden_dims = 512

x = torch.randn(batch_size, num_heads, N, hidden_dims)

