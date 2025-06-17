import torch

mesh = torch.meshgrid(
    torch.arange(0, 10, 1), # [0, 1, 2, ..., 9]
    torch.arange(0, 10, 1), # [0, 1, 2, ..., 9]
    indexing='ij'
)
[2,10,10]
#u,v index have to be int
# mesh = torch.meshgrid(
#     torch.arange(0, u_index, 1/u_index), # [0, 1, 2, ..., 9]
#     torch.arange(0, v_index,1/v_index), # [0, 1, 2, ..., 9]
#     indexing='ij'
# )
print(mesh)  # Should print (10, 10) (10, 10)