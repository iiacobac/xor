import mindspore.nn as nn
import numpy as np
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, Model, context
from mindspore import dataset as ds
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.nn import Accuracy
import mindspore
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


HIDDEN_SIZE = 16
ITERATIONS = 500

def get_xor():
    yield np.array([0,0]).astype(np.float32), np.array([0]).astype(np.float32)
    yield np.array([0,1]).astype(np.float32), np.array([1]).astype(np.float32)
    yield np.array([1,0]).astype(np.float32), np.array([1]).astype(np.float32)
    yield np.array([1,1]).astype(np.float32), np.array([0]).astype(np.float32)

ds_train = ds.GeneratorDataset(list(get_xor()), column_names=['data','label'], column_types=[mindspore.float32, mindspore.float32])
ds_train = ds_train.batch(1)
ds_train = ds_train.repeat(1)

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
optim = nn.Momentum(m.trainable_params(), 0.05, 0.9)

loss = nn.MSELoss()

loss_cb = LossMonitor()

model = Model(m, loss, optim, {'acc': Accuracy()})

time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())

model.train(ITERATIONS, ds_train, callbacks=[time_cb, loss_cb], dataset_sink_mode=False)

print("TF", model.predict(Tensor([[1,0]], mindspore.float32)).asnumpy())
print("FF", model.predict(Tensor([[0,0]], mindspore.float32)).asnumpy())
print("TT", model.predict(Tensor([[1,1]], mindspore.float32)).asnumpy())
print("FT", model.predict(Tensor([[0,1]], mindspore.float32)).asnumpy())
