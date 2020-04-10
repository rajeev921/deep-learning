from prepdata import PrepareData
import matplotlib.pyplot as plt
import pandas as pd  

#Create an instance
obj = PrepareData()

# prepare dataset
train_features, train_targets, val_features, val_targets, test_features, test_targets = obj.getData()

def MSE(y, Y):
	return np.mean((y - Y)**2)

import sys

####################
### Set the hyperparameters in you myanswers.py file ###
####################

from Neural_Network import * 

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
	# Go through a random batch of 128 records from the training dataset
	batch = np.random.choice(train_features.index, size=128)
	X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']

	network.train(X, y)

	# Printing out the training progress
	train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
	val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
	sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
	sys.stdout.flush()

	losses['train'].append(train_loss)
	losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()

obj.plotPredictions(network)
