import numpy as np

# Numpy formatter
# np.set_printoptions(formatter={'float_kind':lambda x: '{0:0.3f}'.format(x)})
np.set_printoptions(precision=4)
np.random.seed(0)   # Set seed

# Learning rate, from 1e-3 to 1e-6 depending on the problem
learning_rate = 1e-2

# Set training data
x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
t_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

# Define weight, W and bias, b as y = Wx + b
W = np.random.rand(1, 1)
b = np.random.rand(1)

# Loss / cost function
def loss_func(x, t):
    y = np.dot(x, W) + b
    return (np.sum( (t - y)**2 )) / (len(x))

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
    y = np.dot(x, W) + b
    return ( np.sum( (t - y)**2 )) / (len(x))

# Predict values after trainning
def predict(x):
    y = np.dot(x, W) + b
    return y

f = lambda x : loss_func(x_data, t_data)

# Print initial values
print('\n---------------------------------------------------')
print('Regression')
print('---------------------------------------------------\n')
print('Initial values')
print('W=', W, ', W.shape=', W.shape, ', b=', b, ', b.shape=', b.shape)
print('Initial error=', round(error_val(x_data, t_data), 4), '\n')

for step in range(8001):
    W = W - learning_rate * derivative(f, W)
    b = b - learning_rate * derivative(f, b)

    if(step % 400 == 0):
        print('step=', step, ', error=', 
        round(error_val(x_data, t_data), 4), ', W=', W, ', b=', b)

print('\n---------------------------------------------------')
p_value = predict(43)
print('predict=', p_value , '\n')