import numpy as np
from network import Network
from num_to_np import prepros



def neuro(a,b):
    p=prepros(a,b)
    neta=Network()
    aa=neta.nue(1,1000,p,np.load("bias1.npy"), np.load("weights1.npy"))
    np.save("first layer.npy",aa)
    netb=Network()
    ab=netb.nue(2,25,aa,np.load("bias2.npy"), np.load("weights2.npy"))
    np.save("third layer.npy", ab)
    netc=Network()
    ac=netc.nue(3,4,ab,np.load("bias3.npy"), np.load("weights3.npy"))
    return ac

