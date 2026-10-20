import torch

a = torch.arange(12)
a2 = a[2].item()
a_2 = a[-2].item()
a_2_4 = a[6:8]
a_2_4[1] = -100
a_3 = a[3:]
a_5 = a[-5:-1:2]
a_2 = a[::2]
a[:4] = torch.IntTensor([-1,-2,-3,-4])
print("a= ", a)
x = torch.tensor([[1,2,3],[10,20,30],[100,200,300]])
print(x[1,1], x[-1,-1], x[0, :], x[:,1])
y =torch.arange(1,82).view(3,3,3,3)
print(y[1,2,0,1], y[:,1,:,:], y[:,:,1, 1])
z = torch.arange(1,9)
w = z[torch.arange(2,6)]
q = z[[0,1,5,7]]
bIndx = torch.tensor([True, True, False, False, False, True, False, False])
s = z[bIndx]
print(w, "\n", q, "\n", s)
i = z > 5
print(i, "\n", z[z>5])