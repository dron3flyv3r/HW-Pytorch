import torch.nn as nn

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.lr1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.lr1(x)
        return x