import torch
import torch.nn as nn
from tqdm import tqdm

# The math is f(x) = x*5
# here is a wary simple ai that will learn to predict the value of f(5), the answer to the question is 10

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

xTest = torch.tensor([[5]], dtype=torch.float32)
nSam, nFea = X.shape
print(nSam, nFea)

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.l1(x)

input_size = nFea
output_size = nFea
model = Net(input_size, output_size)

print(f"Before training: f(5) = {model(xTest).item():.3f}")

# Training loop
lr = 0.005
epochs = 5000

loss = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in tqdm(range(epochs), desc="Training"):
    # Forward pass
    y_pred = model(X)
    
    # Loss
    l = loss(Y, y_pred)
    
    # Gradients = backward pass
    l.backward() # compute gradients
    
    # Update weights
    opt.step()
    
    # Reset gradients
    opt.zero_grad()
    
    '''if epoch % 10 ==0:
        [w, b] = model.parameters()
        print(f"[{epoch}] Loss: {l:.3f} w: {w[0][0].item():.3f}")'''
    
print(f"After training: f(5) = {model(xTest).item():.3f}")