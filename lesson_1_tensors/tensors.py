import torch
import numpy as np

# Various ways to initialize tensors
# 1. from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
# 2. from numpy array
np_array = np.array(data)
x_np = torch.tensor(np_array)
# 3. from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random tensor: \n {x_rand} \n")

# Generate tensors from shape tuple
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


# Attributes of a tensor: shape, data typs, device
tensor = torch.rand(3, 4)
print(f"Shape: {tensor.shape} Data type: {tensor.dtype} Device: {tensor.device}")

# Move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Tensor indexing
tensor = torch.rand(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
print('Last column: ', tensor[:, -1])
tensor[:, 1] = 0
print(tensor)


# Joining tensors: torch.cat, torch.stack
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmetic operations
# Matrix multiplication (y1,y2,y3 will have the same value)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
# print('y1: \n', y1)
# print('y2: \n', y2)
# print('y3: \n', y3)

# Element wise product( z1, z2, z3 will have the same value)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
# print('z1: \n', z1)
# print('z2: \n', z2)
# print('z3: \n', z3)


# Single element tensor
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg), type(agg_item))

# Inplace operations (using _ suffix)
print(tensor, "\n")
tensor.add_(5)
print(tensor)


# Bridge with numpy
# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}  type: {type(t)}")
n = t.numpy()
print(f"n {n}  type: {type(n)}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(t, n)

# numpy array to tensor
np.add(n, 1, out=n)
print(t, n)
