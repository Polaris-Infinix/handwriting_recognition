import os
import cv2
import numpy as np
def to_array(number,no):
    path=os.path.join("dataset","MNIST Dataset JPG format","MNIST - JPG - training",str(int(number)),str(int(no))+".jpg")
    if os.path.exists(path)==True:
        pass
    else:
        while os.path.exists(path)!=True:
            no=no+1
            path = os.path.join("dataset", "MNIST Dataset JPG format", "MNIST - JPG - training", str(int(number)),
                                str(int(no)) + ".jpg")
    image=cv2.imread(path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("fmks",image)
    # cv2.waitKey(0)
    flat=image.flatten()
    flat=flat/255
    np.save("RAM/Layers/1.npy",flat)
    return flat

# to_array(1,10)