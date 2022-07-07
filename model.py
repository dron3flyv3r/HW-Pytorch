import torch.nn as nn



class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.l1(x)