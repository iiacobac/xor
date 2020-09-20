import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

HIDDEN_SIZE = 4
ITERATIONS = 100

tf.keras.backend.set_floatx('float64')

class Net(keras.layers.Layer):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = layers.Dense(HIDDEN_SIZE, activation="tanh")
        self.fc2 = layers.Dense(1)
    
    def call(self, inputs):
        x = self.fc1(inputs)
        return self.fc2(x)

m = Net()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

loss_fn = tf.keras.losses.MeanSquaredError()

for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2
        input = np.array([[1. if x1 else -1., 1. if x2 else -1.]])
        target = np.array([[1. if x1 != x2 else -1.]])
        with tf.GradientTape() as tape:
            output = m(input)
            loss = loss_fn(target, output)

        mloss += loss.numpy()
        gradients = tape.gradient(loss, m.trainable_weights)
        optimizer.apply_gradients(zip(gradients, m.trainable_weights))

    mloss /= 4.
    print("loss: %0.9f" % mloss)

print("TF", m(np.array([[1.,-1.]])).numpy()[0])
print("FF", m(np.array([[-1.,-1.]])).numpy()[0])
print("TT", m(np.array([[1.,1.]])).numpy()[0])
print("FT", m(np.array([[-1.,1.]])).numpy()[0])

