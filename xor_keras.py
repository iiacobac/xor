import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

HIDDEN_SIZE = 4
ITERATIONS = 100

tf.keras.backend.set_floatx('float64')


m = keras.Sequential()
m.add(layers.Dense(HIDDEN_SIZE, activation="tanh"))
m.add(layers.Dense(1))

opt = keras.optimizers.SGD(learning_rate=0.1)

m.compile(optimizer=opt, loss='mse')

x_train = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
y_train = np.array([[-1.],[1.],[1.],[-1.]])

m.fit(x_train, y_train, epochs=ITERATIONS, batch_size=1)

print("TF", m(np.array([[1.,-1.]])).numpy()[0])
print("FF", m(np.array([[-1.,-1.]])).numpy()[0])
print("TT", m(np.array([[1.,1.]])).numpy()[0])
print("FT", m(np.array([[-1.,1.]])).numpy()[0])

