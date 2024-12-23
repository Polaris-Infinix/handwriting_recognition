import numpy as np
import os
import random
from cost import cost
from deltas import delta
from gradients import gradient
from Miscellaneous import no_of_layers
def epocs(no=10):
    Batchsize=64
    if os.path.exists(f"RAM/Epocs/{no}.npy")==False:
        for j in range(no):
            b = np.zeros((Batchsize, 2))
            for i in range(Batchsize):
                ka = random.randint(0, 9)
                kb = random.randint(0, 59000)
                b[i] = [(ka), (kb)]
            np.save(f"RAM/Epocs/{j+1}.npy",b)
    else:
        for i in range(10):
            ep=np.load(f"RAM/Epocs/{i+1}.npy")
            grad2 = 0
            grad3 = 0
            grad4 = 0
            alpha=0.2
            for oq in ep:
                cost(int(oq[0]),int(oq[1]))
                delta()
                gradient()
                grad2=grad2+np.load("RAM/Gradient/gradient2.npy")
                grad3 = grad3 + np.load("RAM/Gradient/gradient3.npy")
                grad4 = grad4 + np.load("RAM/Gradient/gradient4.npy")
            grad4=grad4/Batchsize
            grad3 = grad3 /Batchsize
            grad2 = grad2 /Batchsize
            weight2=np.load("weights/weights2.npy")+alpha*grad2
            weight3 = np.load("weights/weights3.npy") + alpha * grad3
            weight4 = np.load("weights/weights4.npy")+ alpha* grad4
            bias2=np.load("bias/bias2.npy")+alpha*np.load("RAM/Deltas/delta2.npy")
            bias3 = np.load("bias/bias3.npy") + alpha* np.load("RAM/Deltas/delta3.npy")
            bias4 = np.load("bias/bias4.npy") + alpha*np.load("RAM/Deltas/delta-lastlayer.npy")
            np.save("weights/weights2.npy",weight2)
            np.save("weights/weights3.npy", weight3)
            np.save("weights/weights4.npy", weight4)
            np.save("bias/bias2.npy",bias2)
            np.save("bias/bias3.npy", bias3)
            np.save("bias/bias4.npy", bias4)
            print("""First set of 63 batches is complete and its running smoothly
             ******
             ****
             ****
             ***
             *
             ***
             *
             *
             *
             **
             ***
             """)
epocs()
epocs()
epocs()
epocs()
epocs()
epocs()
epocs()
epocs()
epocs()



