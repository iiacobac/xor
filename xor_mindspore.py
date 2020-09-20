import mindspore.nn as nn
import numpy as np
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

HIDDEN_SIZE = 4
ITERATIONS = 100

class Net(nn.Cell):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(HIDDEN_SIZE)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Dense(1)

    def construct(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

m = Net()

optim = nn.SGD(params=m.trainable_params(), learning_rate=0.1)

loss = nn.MSELoss()
m.compile(optimizer=opt, loss='mse')

x_train = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
y_train = np.array([[-1.],[1.],[1.],[-1.]])

m.fit(x_train, y_train, epochs=ITERATIONS, batch_size=1)

print("TF", m(np.array([[1.,-1.]])).numpy()[0])
print("FF", m(np.array([[-1.,-1.]])).numpy()[0])
print("TT", m(np.array([[1.,1.]])).numpy()[0])
print("FT", m(np.array([[-1.,1.]])).numpy()[0])
