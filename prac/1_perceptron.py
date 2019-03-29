'''
Deep learning using Keras
Last Updated : 03/18/2019, by Hyungmin Jun (hyungminjun@outlook.com)

=============================================================================

This Python script with Keras library is an open-source, to implement tutorials
for the machine learning.
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

import keras
import numpy

X = numpy.array([0, 1, 2, 3, 4, 5])
Y = X * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))

# Stochastic gradient descent and mean square error
model.compile("SGD", "mse")
model.fit(X, Y, epochs=1000, verbose=0)

#predict = model.predict(X)
predict = model.predict(X).flatten()

print('Target: ', Y)
print('Predict: ', predict)