import os

def no_of_layers():
    count=0
    for i in range(50):
        path=os.path.join("RAM","Layers",f"{i}.npy")
        if os.path.exists(path)==True:
            count=count+1
    return count

