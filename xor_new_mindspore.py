import mindspore.nn as nn
import numpy as np
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, Model, context
from mindspore import dataset as ds
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.nn import Accuracy
import mindspore
import random
from mindspore.common.initializer import Normal
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU") #, enable_auto_mixed_precision=False, enable_reduce_precision=False)

HIDDEN_SIZE = 4
ITERATIONS = 300

class Net(nn.Cell):

    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Dense(2, hidden_size)
        self.fc2 = nn.Dense(hidden_size, 1)
        self.sig = ops.Sigmoid()

    def construct(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        x = self.fc2(x)
        return x

m = Net(HIDDEN_SIZE)

# create your optimizer
optim = nn.Momentum(m.trainable_params(),learning_rate=0.15, momentum=0.9)

loss_fn = nn.MSELoss()

loss_net = nn.WithLossCell(m, loss_fn)
train_net = nn.TrainOneStepCell(loss_net, optim)
train_net.set_train(True)

for e in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2
        data = Tensor([[1. if x1 else 0., 1. if x2 else 0.]], mindspore.float32)
        label = Tensor([[1. if x1 != x2 else 0.]], mindspore.float32)
        #import pdb;pdb.set_trace()
        loss = train_net(data, label)
        print(f"data: {data}, label: {label}, pred: {m(data)}, loss: %0.9f" % loss.asnumpy())
        mloss += loss.asnumpy()
    mloss /= 4.
    #print("loss: %0.9f" % mloss)

#import pdb;pdb.set_trace()
print("TF", m(Tensor([[1.,0.]], mindspore.float32)).asnumpy())
print("FF", m(Tensor([[0.,0.]], mindspore.float32)).asnumpy())
print("TT", m(Tensor([[1.,1.]], mindspore.float32)).asnumpy())
print("FT", m(Tensor([[0.,1.]], mindspore.float32)).asnumpy())


