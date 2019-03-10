'''
Deep learning for MNIST without back propagation - Machine Learning
Last Updated : 03/10/2019, by Hyungmin Jun (hyungminjun@outlook.com)

=============================================================================

This Python script is an open-source, to implement codes for machine learning
without TensonFlow. The original script comes from NeoWizard.
Copyright 2019 Hyungmin Jun. All rights reserved.

License - GPL version 3
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version. This 
program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Numerical derivative
def derivative(fnc, x):
    del_x = 1e-4    # 0.0001

    grad = np.zeros_like(x)
    it   = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        
        # Save the current value
        tmp_val = x[idx]

        x[idx]    = float(tmp_val) + del_x
        fx1       = fnc(x)
        x[idx]    = float(tmp_val) - del_x
        fx2       = fnc(x)
        grad[idx] = (fx1-fx2) / (2*del_x)
        
        # Recovery the original value
        x[idx] = tmp_val
        
        it.iternext()
    return grad

# Logic gate class
class LogicGate:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):

        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Second hidden layer unit
        self.W2 = np.random.rand(self.input_nodes, self.hidden_nodes)
        self.b2 = np.random.rand(self.hidden_nodes)

        # Output layer unit
        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b3 = np.random.rand(self.output_nodes)

        # Learning rate
        self.learning_rate = 1e-4

    # Cross-entropy from feed forward
    def feed_forward(self):
        delta = 1e-7

        z1 = np.dot(self.input_data, self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1, self.W3) + self.b3
        y  = sigmoid(z2)

        # Cross-entropy
        return -np.sum( self.target_data*np.log(y + delta) + (1 - self.target_data)*np.log((1 - y) + delta) )

    # Calculate errors
    def loss_val(self):
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y  = a3 = sigmoid(z3)

        # Cross-entropy
        return -np.sum( self.__tdata*np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y) + delta) )

    # Train
    def train(self, training_data):
        f = lambda x : self.feed_forward()

        print('Initial error=', self.loss_val())

        for step in range(10001):
            self.__W2 = self.__W2 - self.__learning_rate * derivative(f, self.__W2)
            self.__b2 = self.__b2 - self.__learning_rate * derivative(f, self.__b2)
            self.__W3 = self.__W3 - self.__learning_rate * derivative(f, self.__W3)
            self.__b3 = self.__b3 - self.__learning_rate * derivative(f, self.__b3)
            if(step % 400 == 0):
                print('step=', step, 'error=', self.loss_val())

    # Predict the value
    def predict(self, x_data):
        z2 = np.dot(x_data, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y  = a3 = sigmoid(z3)

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False
        return y, result

# --------------------------------------------------
# MNIST
# --------------------------------------------------
traning_data = np.loadtxt('./data/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data    = np.loadtxt('./data/mnist_test.csv',  delimiter=',', dtype=np.float32)

print('training_data.shape=', traning_data.shape, 'test_data.shape=', test_data.shape)

img = traning_data[0][1:].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()