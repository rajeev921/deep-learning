'''
Predicting Student Admissions with Neural Networks. The goal here is to predict 
if a student will be admitted to a graduate program based on these features. 

The dataset originally came from here: http://www.ats.ucla.edu/

This dataset has three input features: GRE score, GPA, and the rank of the 
undergraduate school (numbered 1 through 4). Institutions with rank 1 have the 
highest prestige, those with rank 4 have the lowest

'''

import PrepareData as prepData

# 8. prepare the data
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
