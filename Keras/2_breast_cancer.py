'''
Breast Cancer Prediction using Keras
Last Updated : 03/19/2019, by Hyungmin Jun (hyungminjun@outlook.com)

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

# Import libraries
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.metrics import confusion_matrix

# Load data and return dataset
def dataLoad():
    dataset = pd.read_csv('./data/breast_cancer_winsconsin.csv')
    print(dataset.head())

    # ID, unnamed column drop, which is not used
    dataset = dataset.drop(['id', 'Unnamed: 32'], axis=1)
    print(dataset.head())
    print(dataset.shape)
    return dataset

# Mapping function to map different string objects to integer
def mapping(data, feature):
    featureMap = dict()
    count = 0
    for i in sorted(data[feature].unique(), reverse=True):
        featureMap[i] = count
        count = count + 1
    data[feature] = data[feature].map(featureMap)
    return data

# Load data
dataset = dataLoad()

# Check whether there is null value
pd.isnull(dataset).sum()

dataset = mapping(dataset, feature="diagnosis")
sample_list = dataset.sample(5)
print(sample_list)