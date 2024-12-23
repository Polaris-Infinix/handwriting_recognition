import numpy as np
import os
from Miscellaneous import no_of_layers
def delta():
    dll=np.load("RAM/Deltas/delta-lastlayer.npy")
    n=no_of_layers()
    for i in range(n):
        path = f"weights/weights{n - i}.npy"
        if os.path.exists(path)==False:
            print("path broken")
            break
        if i>0:
            dll=np.load(f"RAM/Deltas/delta{n-i}.npy")
        w=np.load(f"weights/weights{n-i}.npy")
        l=np.load(f"RAM/Layers/{n-i-1}.npy")
        nn=l.size
        dlln=dll.size
        delt=np.zeros(nn)
        for j in range(nn):
            for k in range(dlln):
                delt[j]=dll[k]*w[j][k]+delt[j]
            delt[j]=delt[j]*l[j]*(1-l[j])
        np.save(f"RAM/Deltas/delta{n-i-1}",delt)





