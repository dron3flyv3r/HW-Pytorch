import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([[4]], dtype=torch.float32)

n_sam, n_fea = X.shape

print(n_sam, n_fea)

class Net(nn.Module):
    """Some Information about Net"""
    """This is a way simpel linear regression model with one input and one output"""
    def __init__(self):
        super(Net, self).__init__()
        self.lr1 = nn.Linear(n_fea, 1)

    def forward(self, x):
        x = self.lr1(x)
        return x
    
model = Net()

loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 5000

pbar = tqdm(range(epochs))

for epoch in pbar:
    # Forward pass & Loss
    y_pred = model(X)
    w = loss(y_pred, y)
    
    # Backward pass
    w.backward()
    
    # Update weights
    optim.step()
    optim.zero_grad()
    
    # See progress
    if epoch % 100 == 0:
        with torch.no_grad():
            y_pred = model(X_test)
            acc = torch.mean((y_pred - y)**2)
            pbar.set_description(f"Loss: {w.item():.8f}, Acc: {acc.item():.2f}")

print(f"After training: f(5) = {model(X_test).item():.3f}")