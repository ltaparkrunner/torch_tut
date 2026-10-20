import torch
t = torch.Tensor(3, 5, 2)
t2 = torch.empty(3, 5, 2, dtype=torch.int32)
print("t2.dtype = ", t2.dtype)
t3 = torch.tensor([[[3,4,5],[3,6,7],[4,8,5],[5,8,3],[4,4,2]], 
                   [[5,5,1],[1,1,3], [4,8,9],[9,3,3],[2,1,1]]], dtype = torch.float32)
print("tensor size = ",t3.size(), "tensor type = ", t3.dtype, "  ", t3.type(), "dimentions = ", t3.dim())
print("tensor.shape = ", t3.shape)
import numpy as np
d_np = np.array([[1,2,3], [4,5,6]])
t4 = torch.from_numpy(d_np)
print("tensor type = ", t4.dtype)
d_np[0,0] = 0
print("t4 = ", t4)
t4[0,1] = 55
print("d_np = ", d_np)
d = t4.numpy().copy()
t4f = t4.float()
t4[0,2] = 12
print("t4f = ", t4f)
t5 = torch.eye(4,4)
print("torch.eye(4,4) = ", t5)
print("torch.full((3,2), 2) = ", torch.full((3,2), 2))
print("torch.arange(-3,5,2) = ", torch.arange(-3,5,2))
print("torch.linspace(-3., 5., 6) = ",torch.linspace(-3., 5., 6))
print(torch.rand(2,3))
print(torch.randn(2,3))
t6 = torch.Tensor(3, 4, 5)
t6.random_(0, 10)
t6.uniform_(0,10)
t6.normal_(0, 1)

t7 = torch.arange(0,12)
t8 = t7.view(3,4)
# t9 = t8.resize(2,3)
t8.resize_(2,3)
t9 = t8.view(3,2)
t10 = t8.ravel()
print("t8 = ", t8, "t9 = ", t9, "t10 = ", t10)
t11 = t8.reshape(1, 6)
print("t11 = ", t11, "t11.permute = ", t11.permute(1,0))
t8.mT
x_test = torch.arange(32).view(8,2,2)

x_test4 = torch.unsqueeze(x_test, dim=0)
print(x_test.size(), x_test4.size())