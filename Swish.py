from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time
# Normalize samples to be between 0 and 1. Required for usage in NumPy
# Samples are 28 by 28
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical(y)

# split samples into train and test groups
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.15, random_state=32)


class DeepNeuralNetwork():

    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        self.params = self.initialization()

    def swish(self, x, derivative=False):
        B = 1
        if derivative:
            return (B*x*self.sigmoid(x)) + (self.sigmoid(x) * (B - (B*x*self.sigmoid(x))))
        return B*x*self.sigmoid(x)

    def sigmoid(self, x, derivative=False):
      if derivative:
          return (np.exp(-x))/((np.exp(-x)+1)**2)
      return 1/(1 + np.exp(-x))

    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        np.random.seed(42)

        params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        '''
            Foward pass feeds data through the various neurons.
            The previous layer's outputs are fed through the next layer's weights and then through
            the activation function before moving on.
        '''

        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.swish(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.swish(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.
        '''
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = output - y_train
        change_w['W3'] = np.dot(error, params['A3'])

        # Calculate W2 update
        error = np.multiply(np.dot(params['W3'].T, error), self.swish(
            params['Z2'], derivative=True))
        change_w['W2'] = np.dot(error, params['A2'])

        # Calculate W1 update
        error = np.multiply(np.dot(params['W2'].T, error), self.swish(
            params['Z1'], derivative=True))
        change_w['W1'] = np.dot(error, params['A1'])

        return change_w

    def update_network_parameters(self, changes_to_w):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''

        for key, value in changes_to_w.items():
            for w_arr in self.params[key]:
                w_arr -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == y)

        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                iteration+1, time.time() - start_time, accuracy
            ))


# 784 nodes: 784 pixels down to 10 nodes for each type of digit
dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)
