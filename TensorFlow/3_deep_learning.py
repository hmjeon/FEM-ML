'''
Deep learning MNIST example - Machine Learning
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
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print('\n', mnist.train.num_examples, mnist.test.num_examples, mnist.validation.num_examples)

print('\ntrain image shape=', np.shape(mnist.train.images))
print('train label shape=',   np.shape(mnist.train.labels))
print('test image shape=',    np.shape(mnist.test.images))
print('test label shape=',    np.shape(mnist.test.labels))

learning_rate = 0.1     # Learning rate
epochs        = 100     # Iteration number
batch_size    = 100     # Batch size

input_nodes  = 784
hidden_nodes = 100
output_nodes = 10

X = tf.placeholder(tf.float32, [None, input_nodes])
T = tf.placeholder(tf.float32, [None, output_nodes])

# Hidden layer unit
W2 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes]))
b2 = tf.Variable(tf.random_normal([hidden_nodes]))

# Output layer unit
W3 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
b3 = tf.Variable(tf.random_normal([output_nodes]))

# Linear regression for the hidden layer
Z2 = tf.matmul(X, W2) + b2
A2 = tf.nn.relu(Z2)             # Relu function

# Linear regression for the output layer
Z3 = logits = tf.matmul(A2, W3) + b3
y  = A3 = tf.nn.softmax(Z3)     # Softmax function

# Cross-entropy
loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=T))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train     = optimizer.minimize(loss)

predicted_val = tf.equal(tf.argmax(A3, 1), tf.argmax(T, 1))

accuracy = tf.reduce_mean(tf.cast(predicted_val, dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(epochs):

        total_batch = int(mnist.train.num_examples / batch_size)

        for step in range(total_batch):

                batch_x_data, batch_t_data = mnist.train.next_batch(batch_size)
                loss_val, _ = sess.run([loss, train], feed_dict={X: batch_x_data, T: batch_t_data})

                if step % 100 == 0:
                    print('step=', step, ', loss_val=', loss_val)        

    # Accuracy
    test_x_data = mnist.test.images
    test_t_data = mnist.test.labels

    accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, T: test_t_data})

    print('\nAcurracy=', accuracy_val)