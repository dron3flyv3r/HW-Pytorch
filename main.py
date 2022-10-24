import torch
import torch.nn as nn
from model import Net
from tqdm import tqdm


lr = 0.001
epochs = 5000

# The math is f(x) = x*2
# here is a wary simple ai that will learn to predict the value of f(5), the answer to the question is 10

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
xTest = torch.tensor([5], dtype=torch.float32)

model = Net()

criterion = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=lr)
#opt = optim.Apollo(model.parameters(), lr=lr)


for __ in tqdm(range(epochs), desc="Training"):
    # Forward pass
    y_pred = model(X)
    
    # Loss
    loss = criterion(Y, y_pred)
    
    # Gradients = backward pass
    loss.backward() # compute gradients
    
    # Update weights
    opt.step()
    
    # Reset gradients
    opt.zero_grad()
    
print(f"After training: f(5) = {model(xTest).item():.3f}")
