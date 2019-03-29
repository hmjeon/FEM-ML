'''
Deep learning for MNIST with back propagation - Machine Learning
Last Updated : 03/11/2019, by Hyungmin Jun (hyungminjun@outlook.com)

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

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Define sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define NeuralNetwork class
class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Second hidden layer unit,  Xavier/He method
        self.W2 = np.random.randn(self.input_nodes, self.hidden_nodes) / np.sqrt(self.input_nodes/2)
        self.b2 = np.random.rand(self.hidden_nodes)      
        
        # Output layer unit, Xavier/He method
        self.W3 = np.random.randn(self.hidden_nodes, self.output_nodes) / np.sqrt(self.hidden_nodes/2)
        self.b3 = np.random.rand(self.output_nodes)      
                        
        # Output layer from linear regression and its output 
        self.Z3 = np.zeros([1, output_nodes])
        self.A3 = np.zeros([1, output_nodes])
        
        # Hidden layer from linear regression and its output 
        self.Z2 = np.zeros([1, hidden_nodes])
        self.A2 = np.zeros([1, hidden_nodes])
        
        # Input layer from linear regression and its output 
        self.Z1 = np.zeros([1, input_nodes])    
        self.A1 = np.zeros([1, input_nodes])       
        
        # learning rate
        self.learning_rate = learning_rate
        
    # Cross-entropy from feed forward
    def feed_forward(self):  
        
        delta = 1e-7
        
        # Input layer - linear regression (Z1) and its output (A1)
        self.Z1 = self.input_data
        self.A1 = self.input_data
        
        # Input layer - linear regression (Z2) and its output (A2)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        # Input layer - linear regression (Z3) and its output (A3)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)
        
        return  -np.sum( self.target_data*np.log(self.A3 + delta) + (1 - self.target_data)*np.log((1 - self.A3) + delta ) )    
    
    # Calculate errors
    def loss_val(self):
        
        delta = 1e-7
        
        # Input layer - linear regression (Z1) and its output (A1)
        self.Z1 = self.input_data
        self.A1 = self.input_data
        
        # Input layer - linear regression (Z2) and its output (A2)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        # Input layer - linear regression (Z3) and its output (A3)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = sigmoid(self.Z3)
        
        return  -np.sum( self.target_data*np.log(self.A3 + delta) + (1-self.target_data)*np.log((1 - self.A3)+delta ) )

    # Train, # of input_data = 784, # of target_data = 10
    def train(self, input_data, target_data):
        
        self.target_data = target_data    
        self.input_data  = input_data
        
        # Feed forward to calculate current errors
        loss_val = self.feed_forward()
        
        # Output layer - loss_3
        loss_3  = (self.A3 - self.target_data) * self.A3 * (1 - self.A3)
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)   
        self.b3 = self.b3 - self.learning_rate * loss_3
        
        # Hidden layer - loss_2        
        loss_2  = np.dot(loss_3, self.W3.T) * self.A2 * (1-self.A2)
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)   
        self.b2 = self.b2 - self.learning_rate * loss_2

    # Accuracy
    def accuracy(self, test_data):
        
        matched_list     = []
        not_matched_list = []
        
        for index in range(len(test_data)):
                        
            label = int(test_data[index, 0])
                        
            # Normalize data for one-hot encoding
            data = (test_data[index, 1:] / 255.0 * 0.99) + 0.01

            predicted_num = self.predict(np.array(data, ndmin = 2))
        
            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
                
        print("Current Accuracy=", 100*(len(matched_list)/(len(test_data))), '%')
        
        return matched_list, not_matched_list

    # Predict the value
    def predict(self, input_data):   
        
        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = sigmoid(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = sigmoid(Z3)
        
        predicted_num = np.argmax(A3)

        return predicted_num

# --------------------------------------------------
# MNIST
# --------------------------------------------------
training_data = np.loadtxt('./Data/mnist_train.csv', delimiter = ',', dtype = np.float32)
test_data     = np.loadtxt('./Data/mnist_test.csv',  delimiter = ',', dtype = np.float32)

input_nodes   = 784
hidden_nodes  = 100
output_nodes  = 10
learning_rate = 0.3
epochs        = 1

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Set timer
s_time = datetime.now()
for i in range(epochs):
    
    for step in range(len(training_data)):
        
        # Normalize data
        target_data = np.zeros(output_nodes) + 0.01    
        target_data[int(training_data[step, 0])] = 0.99

        input_data = ((training_data[step, 1:] / 255.0) * 0.99) + 0.01

        nn.train( np.array(input_data, ndmin = 2), np.array(target_data, ndmin = 2) )

        if step % 400 == 0:
            print("step=", step, ", loss_val=", nn.loss_val())

e_time = datetime.now()
print("\nTraining time=", e_time - s_time)

# Accuracy
nn.accuracy(test_data)