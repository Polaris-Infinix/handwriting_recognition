from network import Neuron
from image_processing import to_array
import numpy as np
def net(a,b):
    aa=Neuron()
    layer2 = aa.neu(1,200, to_array(a, b),np.load("bias/bias2.npy"),np.load("weights/weights2.npy"))
    layer3 = aa.neu(2,100, layer2,np.load("bias/bias3.npy"),np.load("weights/weights3.npy"))
    layer4 = aa.neu(3, 10, layer3,np.load("bias/bias4.npy"),np.load("weights/weights4.npy"))
    print("Layer 4 is the output layer")
    return layer4


# print(net(4,80))
