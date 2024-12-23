import torch
from torch import nn
import matplotlib.pyplot as plt
print(torch.__version__)
X=torch.load("X_tensors.pt")
Y=torch.load("Y_tensors.pt")
A=X[1]
B=Y[1]
i=2
def to_out(num):
  l=torch.zeros(1,10)
  l[0][num]=1
  return l

while i<=10:
    A=torch.concat((A,X[i]),dim=0)
    B=torch.concat((B,Y[i]),dim=0)
    i+=1

print(B.size())
print(B)
T=to_out(int(B[1][0].item()))
print(T)
i=1
while i<640:
    H=to_out(int(B[i][0].item()))
    T=torch.concat((T,H),dim=0)
    i+=1
print(T.size())
torch.save(T,"Y.pt")
torch.save(A,"X.pt")
print(A.size())

#
