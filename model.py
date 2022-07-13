import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.lr1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.lr1(x)
        return x