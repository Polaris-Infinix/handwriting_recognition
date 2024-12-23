import numpy as np
import activation as sig

class Neuron:
    def neu(self,h,no,input,bias=None,weight=None):
        o=input.size
        if bias is None:
            bias=np.zeros(no)
            np.save(f"bias/bias{h + 1}.npy", bias)
        if weight is None:
            weight = np.random.randn(o, no) * np.sqrt(1. / o)
            np.save(f"weights/weights{h + 1}.npy", weight)
        output=np.zeros(no)
        i=0

        while i<no:
            j=0
            while j<o:
                output[i]=input[j]*weight[j,i]+output[i]
                j=j+1
            output[i]=output[i]+bias[i]
            output[i]=sig.sigmoid(output[i])
            i=i+1

        print(f"Execution of {h+1} layer is complete")
        np.save(f"RAM/Layers/{h+1}.npy",output)
        return output

