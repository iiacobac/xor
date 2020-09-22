import mindspore.nn as nn
import numpy as np
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, Model, context
from mindspore import dataset as ds
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.nn import Accuracy
import mindspore 

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


HIDDEN_SIZE = 4
ITERATIONS = 1000

def get_xor():
    yield np.array([0,0]).astype(np.float32), np.array([0,1]).astype(np.float32)
    yield np.array([0,1]).astype(np.float32), np.array([1,0]).astype(np.float32)
    yield np.array([1,0]).astype(np.float32), np.array([1,0]).astype(np.float32)
    yield np.array([1,1]).astype(np.float32), np.array([0,1]).astype(np.float32)

ds_train = ds.GeneratorDataset(list(get_xor()), column_names=['data','label'])
ds_train = ds_train.batch(4)
ds_train = ds_train.repeat(1)

class Net(nn.Cell):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(2,HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(HIDDEN_SIZE,2)

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

m = Net()
optim = nn.Momentum(m.trainable_params(), 0.1, 0.9)

loss = nn.SoftmaxCrossEntropyWithLogits()

loss_cb = LossMonitor()

model = Model(m, loss, optim, {'acc': Accuracy()})

time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
 
model.train(ITERATIONS, ds_train, callbacks=[time_cb, loss_cb], dataset_sink_mode=False)


print("TF", model.predict(Tensor([[1,0]], mindspore.float32)).asnumpy())
print("FF", model.predict(Tensor([[0,0]], mindspore.float32)).asnumpy())
print("TT", model.predict(Tensor([[1,1]], mindspore.float32)).asnumpy())
print("FT", model.predict(Tensor([[0,1]], mindspore.float32)).asnumpy())


