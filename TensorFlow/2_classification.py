'''
Logical regression - Machine Learning
Last Updated : 03/12/2019, by Hyungmin Jun (hyungminjun@outlook.com)

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

import tensorflow as tf
import numpy as np

loaded_data = np.loadtxt('./Data/data_diabetes.csv', delimiter=',')

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print('x_data.shape=', x_data.shape)
print('t_data.shape=', t_data.shape)

W = tf.Variable(tf.random_normal([8, 1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32, [None, 8])
T = tf.placeholder(tf.float32, [None, 1])

# Linear regression
z = tf.matmul(X, W) + b

# Sigmoid
y = tf.sigmoid(z)

# Cross-entropy
loss = -tf.reduce_mean( T*tf.log(y) + (1-T)*tf.log(1-y) )

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train     = optimizer.minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float32)
accuracy  = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(20001):

        loss_val, _ = sess.run([loss, train], feed_dict={X: x_data, T: t_data})

        if step % 500 == 0:
            print('step=', step, ', loss_val=', loss_val)

    # Accuracy
    y_val, predicted_val, accuracy_val = sess.run([y, predicted, accuracy], feed_dict={X: x_data, T: t_data})

    print('\ny_val.shape=', y_val.shape, ', predicted_val=', predicted_val.shape)
    print('\nAcurracy=', accuracy_val)