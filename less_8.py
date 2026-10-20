import torch
print("version = ", torch.__version__)

a = torch.arange(1,10).view(3,3)
b = torch.arange(10,19).view(3,3)

mul1 = a*b
c = torch.matmul(a, b)

print('\n', mul1, '\n\n', a.mm(b))

bx = torch.randn(7,3,5)
by = torch.randn(7,5,4)

bc = torch.bmm(bx, by)
print(bc)
a = torch.arange(1,10,dtype=torch.float32)
b = torch.ones(9)

print(a @ b, torch.outer(a,b))
a = torch.tensor([1.,2.,3.])
b = torch.arange(4., 10.).view(2,3)
print(b.mv(a) )
a = torch.tensor([[1,2,3], [1,4,9], [1,8,27]])
y = torch.tensor([10, 20, 30])
invA = torch.linalg.inv(a)
print(torch.linalg.matrix_rank(a), torch.linalg.solve(a,y), torch.mv(invA, y))

