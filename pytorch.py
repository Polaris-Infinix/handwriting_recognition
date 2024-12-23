import numpy as np
import torch
import numpy
from image_processing import to_array
Tensorsx={}
Tensorsy={}
k=1
while k<=10:
    OP = numpy.load(f"RAM/Epocs/{k}.npy")
    j = 0
    for i in OP:
        X_train_num = to_array(i[0], i[1])
        Y_train_num=np.array(i[0])
        if j == 0:
            X = torch.from_numpy(X_train_num)
            X = X.unsqueeze(0)  # Makes it [1, 784]
            Y=torch.from_numpy(Y_train_num)
            Y=Y.unsqueeze(0) #Makes it [1,1]
            Y = Y.unsqueeze(0)
            print(X, "Hello")
            j = 1
            continue

        x = torch.from_numpy(X_train_num)
        x = x.unsqueeze(0)  # Makes it [1, 784]
        X = torch.cat((X, x), dim=0)  # This will maintain [N, 784] shape
        y=torch.from_numpy(Y_train_num)
        y=y.unsqueeze(0) #Makes it [1,1]
        y = y.unsqueeze(0)
        Y=torch.cat((Y,y),dim=0)
        print(X.size(),Y.size())
    Tensorsx[k]=X
    Tensorsy[k]=Y
    k+=1
print(Tensorsx)
print(Tensorsy)
torch.save(Tensorsx, 'X_tensors.pt')
torch.save(Tensorsy,"Y_tensors.pt")

