'''
Predicting Student Admissions with Neural Networks
The dataset originally came from here: http://www.ats.ucla.edu/

Loading the data
To load the data and format it nicely, we will use two very useful packages called Pandas and Numpy. You can read on the documentation here:

https://pandas.pydata.org/pandas-docs/stable/
https://docs.scipy.org/
'''

import PrepareData as prepData

features, targets, features_test, targets_test = prepData.splitTrainTestData()

#################### Defining the model architecture ####################
# Here's where we use Keras to build our neural network.

# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(6,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#################### Training the model ####################
# Training the model
model.fit(features, targets, epochs=200, batch_size=100, verbose=0)

#################### Scoring the model #################### 
# Evaluating the model on the training and testing set
score = model.evaluate(features, targets)
print("\n Training Accuracy:", score[1])
score = model.evaluate(features_test, targets_test)
print("\n Testing Accuracy:", score[1])
