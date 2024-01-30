from torch_scatter import scatter_mean
import torch

src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])


out = scatter_mean(src, index, dim=1)

print(out)