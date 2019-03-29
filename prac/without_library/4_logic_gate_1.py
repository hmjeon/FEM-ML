'''
Logic Gate using the classification - Machine Learning
Last Updated : 03/09/2019, by Hyungmin Jun (hyungminjun@outlook.com)

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
    return 1 / (1+np.exp(-x))

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

        # Initialize weight and bias
        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        # Learning rate
        self.__learning_rate = 1e-2

    # Loss / cost function
    def __loss_func(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        # Cross-entropy
        return -np.sum( self.__tdata*np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y) + delta) )

    # Calculate errors
    def error_val(self):
        delta = 1e-7

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)

        # Cross-entropy
        return -np.sum( self.__tdata*np.log(y + delta) + (1 - self.__tdata)*np.log((1 - y) + delta) )

    # Train
    def train(self):
        f = lambda x : self.__loss_func()

        print('Initial error=', self.error_val())

        for step in range(8001):
            self.__W = self.__W - self.__learning_rate * derivative(f, self.__W)
            self.__b = self.__b - self.__learning_rate * derivative(f, self.__b)
            if(step % 400 == 0):
                print('step=', step, 'error=', self.error_val())

    # Predict the value
    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)

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
for input_data in test_data:
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print(input_data, '=', logical_val)
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
for input_data in test_data:
    (sigmoid_val, logical_val) = OR_obj.predict(input_data)
    print(input_data, '=', logical_val)
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
for input_data in test_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(input_data)
    print(input_data, '=', logical_val)
print('\n')

# --------------------------------------------------
# XOR gate - need multi-layered training
# --------------------------------------------------
xdata = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
tdata = np.array([0, 1, 1, 0])

XOR_obj = LogicGate('XOR_GATE', xdata, tdata)
XOR_obj.train()

# XOR gate prediction
print('\n', XOR_obj.name)
test_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
for input_data in test_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(input_data)
    print(input_data, '=', logical_val)
print('\n')