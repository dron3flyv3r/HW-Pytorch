import torch.nn as nn

class Net(nn.Module):
    """Some Information about Net"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.lr1 = nn.Linear(input_size, hidden_size)
        self.lr2 = nn.Linear(hidden_size, hidden_size)
        self.lr3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lr1(x)
        x = self.lr2(self.relu(x))
        x = self.lr3(self.relu(x))
        return x