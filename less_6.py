import torch
print("version = ", torch.__version__)

a = torch.tensor([1., 2., 3.], dtype = torch.float32)
print(a-3, a*5-3)
b = torch.IntTensor([3,4,5])
c = torch.tensor([7, 8, 9], dtype = torch.int16)
d = torch.FloatTensor([10, 11, 12])
print(b, b-a, c, '\n', d, b/a)
e = torch.tensor([[13,14,22],[72,32,53]])
print(e/d, '\n', e+d)
a = torch.arange(1,19).view(3,3,2)
b = torch.ones(3, 2)
print(a-b)