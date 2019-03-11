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

# Download train and test data for MNIST
# http://www.pjreddie.com/media/files/mnist_train.csv
# http://www.pjreddie.com/media/files/mnist_test.csv

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

# NeuralNetwork class
class NeuralNetwork:

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

        z1 = np.dot(self.input_data, self.W2) + self.b2
        y1 = sigmoid(z1)

        z2 = np.dot(y1, self.W3) + self.b3
        y  = sigmoid(z2)

        # Cross-entropy
        return -np.sum( self.target_data*np.log(y + delta) + (1 - self.target_data)*np.log((1 - y) + delta) )

    # Train, input_data = 784, target_data = 10
    def train(self, training_data):

        # Normalize
        self.target_data = np.zeros(output_nodes) + 0.01
        self.target_data[int(training_data[0])] = 0.99

        self.input_data = (training_data[1:] / 255.0 * 0.99) + 0.01

        f = lambda x : self.feed_forward()

        self.W2 = self.W2 - self.learning_rate * derivative(f, self.W2)
        self.b2 = self.b2 - self.learning_rate * derivative(f, self.b2)
        self.W3 = self.W3 - self.learning_rate * derivative(f, self.W3)
        self.b3 = self.b3 - self.learning_rate * derivative(f, self.b3)

    # Predict the value
    def predict(self, input_data):
        
        z1 = np.dot(input_data, self.W2) + self.b2
        y1 = sigmoid(z1)
        
        z2 = np.dot(y1, self.W3) + self.b3
        y  = sigmoid(z2)

        predicted_num = np.argmax(y)
        return predicted_num

    # Accuracy
    def accuracy(self, test_data):

        matched_list     = []
        not_matched_list = []

        for index in range(len(test_data)):

            label = int(test_data[index, 0])

            # Normalize
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01

            predicted_num = self.predict(data)

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        
        print("Current accuracy=", 100*(len(matched_list)/(len(test_data))), ' %')

        return matched_list, not_matched_list

# --------------------------------------------------
# MNIST
# --------------------------------------------------
training_data = np.loadtxt('./data/mnist_train.csv', delimiter=',', dtype=np.float32)
test_data     = np.loadtxt('./data/mnist_test.csv',  delimiter=',', dtype=np.float32)
'''
img = training_data[0][1:].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
'''
input_nodes  = 784
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)

# Training Neural network
for step in range(30001):

    index = np.random.randint(0, len(training_data) - 1)
    nn.train(training_data[index])

    #if step % 400 == 0:
    print('step=', step, ', loss_val', nn.loss_val())

# Accuracy
nn.accuracy(test_data)