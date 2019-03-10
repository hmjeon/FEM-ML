'''
Logistic regression - classification - Machine Learning
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

# Numpy formatter
# np.set_printoptions(formatter={'float_kind':lambda x: '{0:0.3f}'.format(x)})
np.set_printoptions(precision=4)
np.random.seed(0)   # Set seed

# Learning rate, from 1e-3 to 1e-6 depending on the problem
learning_rate = 1e-2

'''
# Set training data
x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0,  1,  1,  1,  1,  1 ]).reshape(10, 1)

# Define weight, W and bias, b as y = Wx + b
W = np.random.rand(1, 1)
b = np.random.rand(1)
'''

# Set training data (multiple variables)
x_data = np.array([[2, 4], [4, 11], [6, 6], [8, 5], [10, 7], [12, 16], [14, 8], 
[16, 3], [18, 7]])
t_data = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(9, 1)

# Define weight, W and bias, b as y = Wx + b
W = np.random.rand(2, 1)
b = np.random.rand(1)

# Define sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))

# Loss / cost function
def loss_func(x, t):
    delta = 1e-7    # To prevent numerical errors

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # Cross-entropy
    return -np.sum( t*np.log(y + delta) + (1 - t)*np.log((1 - y) + delta) )

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

# Loss / cost function
def error_val(x, t):
    delta = 1e-7    # To prevent numerical errors

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # Cross-entropy
    return -np.sum( t*np.log(y + delta) + (1 - t)*np.log((1 - y) + delta) )

# Predict values after trainning
def predict(x):

    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > 0.5:
        result = 1  # True
    else:
        result = 0  # False

    return y, result

f = lambda x : loss_func(x_data, t_data)

# Print initial values
print('\n---------------------------------------------------')
print('Regression')
print('---------------------------------------------------\n')
print('Initial values')
print('W=', W.transpose(), ', W.shape=', W.shape, ', b=', b, ', b.shape=', b.shape)
print('Initial error=', round(error_val(x_data, t_data), 4), '\n')

for step in range(10001):
    W = W - learning_rate * derivative(f, W)
    b = b - learning_rate * derivative(f, b)

    if(step % 400 == 0):
        print('step=', step, ', error=', 
        round(error_val(x_data, t_data), 4), ', W=', W.transpose(), ', b=', b)

print('\n---------------------------------------------------')

'''
test_data = 16
(real_val, logical_val) = predict(test_data)
print('real_val=', real_val, ', logical_val=', logical_val, '\n')
'''

test_data = np.array([3, 17])
(real_val, logical_val) = predict(test_data)
print('real_val=', real_val, ', logical_val=', logical_val, '\n')

test_data = np.array([7, 21])
(real_val, logical_val) = predict(test_data)
print('real_val=', real_val, ', logical_val=', logical_val, '\n')