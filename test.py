import torch
import torch.nn as nn
import torch_optimizer as optim
from tqdm import tqdm

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[6.57],[13.14],[19.71],[26.28]], dtype=torch.float32)

xTest = torch.tensor([[5]], dtype=torch.float32)

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.lr1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.lr1(x)
        return x
    
nSam, nFea = X.shape
print(nSam, nFea)

input_size = nFea
output_size = nFea
model = Net(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
#optimizer = optim.Apollo(model.parameters(), lr=1e-2)

pbar = tqdm(range(5000))

for _ in pbar:
    # Forward pass
    y_pre = model(X)
    
    # Compute loss
    loss = criterion(y, y_pre)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    if _ % 100 == 0:
        pbar.set_description(f"Loss: {loss.item():.8f}")
        
with torch.no_grad():
    y_pred = model(xTest)
    print(f"f(5) = {y_pred.item():.3f}")
    