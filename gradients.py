import numpy as np
from Miscellaneous import no_of_layers
import os
def gradient():
    n=no_of_layers()
    lld=np.load("RAM/Deltas/delta-lastlayer.npy")
    ll=np.load(f"RAM/Layers/{n-1}.npy") 2

    for k in range(n):
        if os.path.exists(f"RAM/Layers/{n-k-1}.npy")==False:
            print("path broken")
            break
        if k==0:
            pass
        elif k!=0:
            lld=np.load(f"RAM/Deltas/delta{n-k}.npy")
            ll=np.load(f"RAM/Layers/{n-k-1}.npy")
        gradientw = np.zeros((ll.size, lld.size))
        for i in range(ll.size):
            for j in range(lld.size):
                gradientw[i][j]=lld[j]*ll[i]
        np.save(f"RAM/Gradient/gradient{n-k}.npy",gradientw)


