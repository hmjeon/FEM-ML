'''
Static beam using CNN model
Last Updated : 03/26/2019, by Hyungmin Jun (hyungminjun@outlook.com)

=============================================================================

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


import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from numpy.linalg import inv

# Get external force vector
def get_vec_ext():
    load = 1000*(np.random.random(n_nde)*2-1)

    # Force vector
    R = np.zeros(n_dof)
    R[0::2] = load
    return R[2:]

# Set train model
def set_train_model():
    i_dim = n_nde - 1
    o_dim = n_nde
    m_dim = 50

    model = Sequential()

    model.add(Dense(m_dim, input_dim=i_dim, activation='tanh',   kernel_initializer='normal'))
    model.add(Dense(m_dim, input_dim=m_dim, activation='tanh',   kernel_initializer='normal'))
    model.add(Dense(o_dim, input_dim=m_dim, activation='linear', kernel_initializer='normal'))

    model.compile(loss='mse', optimizer="adam")
    return model

np.random.seed(1)

n_nde = 100
n_dof = n_nde*2
n_ele = n_nde-1

x = np.linspace(0, 1, n_nde)

L   = 1000          # mm
rho = 0.006         # kg/mm
E   = 210000        # N/mm2
I   = 0.801*10**6   # mm4
A   = 764           # mm^2

# Element stiffness matrix
L_e = L / n_ele
K_e = E*I/L_e**3 * np.matrix([
    [  12,     -6*L_e,    -12,     -6*L_e],
    [  -6*L_e,  4*L_e**2,   6*L_e,  2*L_e**2],
    [ -12,      6*L_e,     12,      6*L_e],
    [  -6*L_e,  2*L_e**2,   6*L_e,  4*L_e**2]])

# Assemble stiffness matrix
n_dof = n_nde * 2
K_mat = np.zeros([n_dof, n_dof])
for i in range(n_ele):
    K_mat[2*i:2*i+4, 2*i:2*i+4] += K_e

# Impose boundary conditions
K_mat = K_mat[2:, 2:]

# Inverse stiffness matrix
K_inv = inv(K_mat)

# Make force vector set
n_set = 10000
U_set = np.zeros([n_set, n_nde])
R_set = np.zeros([n_set, n_nde - 1])

for i in range(n_set):
    R = get_vec_ext()
    U = np.matmul(K_inv, R)

    v = U[0::2]
    v = np.insert(v, 0, 0)
    U_set[i, :] = v
    R_set[i, :] = R[0::2]

# Set train and test set with the ratio 8:2
x_train = R_set[0:int(n_set*0.8), :]*1/10000
y_train = U_set[0:int(n_set*0.8), :]
x_test  = R_set[int(n_set*0.8):, :]*1/10000
y_test  = U_set[int(n_set*0.8):, :]

# Training the model
model = set_train_model()
model.fit(x_train, y_train, epochs=50, verbose = 1)

predictions = model.predict(x_test)

plt.figure(figsize=(8, 5))
plt.plot(x, np.transpose(predictions[0:10,:]))
plt.plot(x, np.transpose(y_test[0:10,:]))
plt.show()

# ==================================================
# Predict
# ==================================================

# Define point load
F     = np.zeros(n_dof - 2)
F[-2] = -10000

# Find reference solution
d = np.matmul(inv(K_mat), F)
dz = d[0::2]
dz = np.insert(dz, 0, 0)

f = F[0::2]*1/10000
pred = model.predict(np.reshape(f, (1,len(f))))

plt.figure(figsize=(8, 5))

plt.plot(x, pred[0])
plt.plot(x, dz)

plt.show()