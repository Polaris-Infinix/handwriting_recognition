import torch
from torch import nn
from image_processing import to_array
# Load the entire model

# Redefine the DigitsRecog class
class DigitsRecog(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 800)
        self.linear2 = nn.Linear(800, 400)
        self.linear3 = nn.Linear(400, 100)
        self.linear4 = nn.Linear(100, 25)
        self.linear5 = nn.Linear(25, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))  # Sigmoid for one-hot encoded outputs
        return x

# Now load the model
model = torch.load("digits_recog_model.pth")
model.eval()

X=torch.from_numpy(to_array(8,800)).to("cuda").float()
print(X.unsqueeze(0))
print(X.size())
Y=model(X)
max=torch.max(Y)
for i in range(10):
    if Y[i]==max:
        print(i)
        break
