import numpy as np
from Network2 import net
def cost(a,b):
    k=np.zeros(9)
    m=np.insert(k,a,1)
    l=net(a,b)
    r=m-l
    delta=(r*2)*(l*(1-l))
    r=r*r
    # print(r)
    print(f"cost={r.sum()}")
    np.save("RAM/Deltas/delta-lastlayer.npy",delta)
    return r

# cost(1,10)