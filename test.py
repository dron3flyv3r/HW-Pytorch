import torch
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28
hidden_size = 64
num_classes = 10
epochs = 10
batch = 64
learning_rate = 0.001

traning_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(traning_dataset, batch_size=batch, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

exsampels = iter(trainloader).next()
samples, label = exsampels

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

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

def acc():
    with torch.no_grad():
        nCorrect = 0
        nSampels = 0
        for inputs, labels in testloader:
            inputs = inputs.reshape(-1, 28*28)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(inputs)
            
            _, predistion = torch.max(outputs, 1)
            
            nSampels += labels.shape[0]
            nCorrect += (predistion == labels).sum().item()
        acc = 100.0 * nCorrect / nSampels
    return acc
    
net = Net(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

pbar = tqdm(range(epochs), unit='epoch')

# loop over the dataset multiple times
for epoch in pbar:
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.reshape(-1, 28*28)
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    pbar.set_description(f"Running loss: {running_loss:.5}, Loss = {loss.item():.8}, acc: {acc()}")

print('Finished Training')
print(f"Total loss: {running_loss:.5}, acc: {acc()}")