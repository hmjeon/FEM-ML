'''
Logic Gate using deep learning - Machine Learning
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
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name

        # Initialize input data
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # Second hidden layer unit
        self.__W2 = np.random.rand(2, 6)
        self.__b2 = np.random.rand(6)

        # Output layer unit
        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        # Learning rate
        self.__learning_rate = 1e-2

    # Cross-entropy from feed forward
    def feed_forward(self):
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y  = sigmoid(z3)

        # Cross-entropy
        return -np.sum( self.__tdata*np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y) + delta) )

    # Calculate errors
    def loss_val(self):
        delta = 1e-7

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y  = sigmoid(z3)

        # Cross-entropy
        return -np.sum( self.__tdata*np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y) + delta) )

    # Train
    def train(self):
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
        y  = sigmoid(z3)

        if y > 0.5:
            result = 1  # True
        else:
            result = 0  # False
        return y, result

# --------------------------------------------------
# AND gate
# --------------------------------------------------
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 0, 0, 1])

AND_obj = LogicGate('AND_GATE', xdata, tdata)
AND_obj.train()

# AND gate prediction
print('\n', AND_obj.name)
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
for data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(data)
    print(data, '=', logical_val, sigmoid_val)
print('\n')

# --------------------------------------------------
# OR gate
# --------------------------------------------------
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 1])

OR_obj = LogicGate('OR_GATE', xdata, tdata)
OR_obj.train()

# OR gate prediction
print('\n', OR_obj.name)
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
for data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(data)
    print(data, '=', logical_val, sigmoid_val)
print('\n')

# --------------------------------------------------
# NAND gate
# --------------------------------------------------
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([1, 1, 1, 0])

NAND_obj = LogicGate('NAND_GATE', xdata, tdata)
NAND_obj.train()

# NAND gate prediction
print('\n', NAND_obj.name)
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
for data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(data)
    print(data, '=', logical_val, sigmoid_val)
print('\n')

# --------------------------------------------------
# XOR gate - need multi-layered training
# --------------------------------------------------
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

XOR = LogicGate('XOR_GATE', xdata, tdata)
XOR.train()

# XOR gate prediction
print('\n', XOR.name)
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
for data in test_data:
    (sigmoid_val, logical_val) = XOR.predict(data)
    print(data, '=', logical_val, sigmoid_val)
print('\n')