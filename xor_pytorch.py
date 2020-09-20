import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_SIZE = 4
ITERATIONS = 100

class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

m = Net(HIDDEN_SIZE)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()


# create your optimizer
optimizer = optim.SGD(m.parameters(), lr=0.1)

for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2
        input = torch.tensor([1. if x1 else -1., 1. if x2 else -1.])
        target = torch.tensor([1. if x1 != x2 else -1.])
        optimizer.zero_grad()   # zero the gradient buffers
        output = m(input)
#        print(input, target, output)
        #import pdb; pdb.set_trace()
        loss = criterion(output, target)
        mloss += loss.item()
        loss.backward()
        optimizer.step()    # Does the update        
    mloss /= 4.
    print("loss: %0.9f" % mloss)

print("TF", m(torch.tensor([1.,-1.])).item())
print("FF", m(torch.tensor([-1.,-1.])).item())
print("TT", m(torch.tensor([1.,1.])).item())
print("FT", m(torch.tensor([-1.,1.])).item())

