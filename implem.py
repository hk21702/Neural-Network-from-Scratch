from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Normalize samples to be between 0 and 1. Required for usage in NumPy
# Samples are 28 by 28
x, y = fetch_openml('mnist_784', version=1, return_X_Y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

# split samples into train and test groups
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=25)


class DeepNeuralNetwork():

    def init(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self. l_rate = l_rate

        self.params = self.initialization()


dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
