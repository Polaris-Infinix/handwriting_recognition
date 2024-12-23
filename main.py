import torch
from torch import nn
import matplotlib.pyplot as plt
X = torch.load('X.pt')
Y=torch.load("Y.pt")
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Assuming X_train and y_train are already loaded
# X_train shape: (batch_size, 1, 28, 28)
# y_train shape: (batch_size, 10) for one-hot encoding

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model
class DigitsRecog(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 800)
        self.linear2 = nn.Linear(800, 400)
        self.linear3 = nn.Linear(400, 100)
        self.linear4 = nn.Linear(100, 25)
        self.linear5 = nn.Linear(25, 10)  # 10 output classes
        self.relu = nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))  # Sigmoid for multi-label classification (one-hot encoded labels)
        return x

# Instantiate the model and move to device
model = DigitsRecog().to(device)

# Loss and optimizer
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for one-hot encoded labels
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# DataLoader for batching
train_dataset = TensorDataset(X_train.float(), y_train.float())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
epochs = 6000
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Flatten the input tensor
        batch_X = batch_X.view(batch_X.size(0), -1).to(device)  # Flatten and move to device
        batch_y = batch_y.to(device)  # Move labels to device

        # Forward pass
        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Save the entire model
torch.save(model, "digits_recog_model1.pth")




