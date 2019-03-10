import numpy as np

def derivative(fnc, x):
    del_x = 1e-4
    grad  = np.zeros_like(x)
    
    it  = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        
        tmp_val = x[idx]
        x[idx]  = float(tmp_val) + del_x
        fx1     = fnc(x)
        
        x[idx]  = tmp_val - del_x
        fx2     = fnc(x)
        
        grad[idx] = (fx1-fx2) / (2*del_x)
        
        x[idx] = tmp_val
        
        it.iternext()
    return grad

def func1(input_obj):
    w = input_obj[0, 0]
    x = input_obj[0, 1]
    y = input_obj[1, 0]
    z = input_obj[1, 1]
    
    return (w*x + x* y *z + 3*w+ z*np.power(y,2))

input = np.array([[1.0, 2.0], [3.0, 4.0]])
print (derivative(func1, input))