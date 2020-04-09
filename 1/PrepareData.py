############ Loading the data ################
# Importing pandas and numpy
import pandas as pd
import numpy as np
import keras

def readCSV():
	# Reading the csv file into a pandas DataFrame
	data = pd.read_csv('student_data.csv')
	# Printing out the first 10 rows of our data
	print(data[:10])

	return data

############ Plotting the data ################
import matplotlib.pyplot as plt

# function to help us plot
def plot_points(data):
	X = np.array(data[['gre', 'gpa']])
	y = np.array(data['admit'])
	#plt.plot(X,y, label='Loaded from file!')
	admitted = X[np.argwhere(y==1)]
	rejected = X[np.argwhere(y==0)]
	plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
	plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
	plt.xlabel('Test (GRE)')
	plt.ylabel('Grades (GPA)')

	plt.legend()
	plt.show()

def data_rank(data):
	'''
	Roughly, it looks like the students with high scores in the grades and test passed, while the ones with low scores didn't, 
	but the data is not as nicely separable as we hoped it would. Maybe it would help to take the rank into account? Let's make 
	4 plots, each one for each rank.
	'''

	# Separating the ranks
	data_rank1 = data[data["rank"]==1]
	data_rank2 = data[data["rank"]==2]
	data_rank3 = data[data["rank"]==3]
	data_rank4 = data[data["rank"]==4]

	# Plotting the graphs
	plot_points(data_rank1)
	plt.title("Rank 1")
	plt.show()
	plot_points(data_rank2)
	plt.title("Rank 2")
	plt.show()
	plot_points(data_rank3)
	plt.title("Rank 3")
	plt.show()
	plot_points(data_rank4)
	plt.title("Rank 4")
	plt.show()

def one_hot_coding(data):
	############## One-hot encoding the rank ################

	# TODO:  Make dummy variables for rank
	one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

	# TODO: Drop the previous rank column
	one_hot_data = one_hot_data.drop('rank', axis=1)

	# Print the first 10 rows of our data
	print("one_hot_data")
	print(one_hot_data[:10])

	return one_hot_data

############## Scaling the data ################
'''
The next step is to scale the data. We notice that the range for grades is 1.0-4.0, whereas the range for test scores is roughly 
200-800, which is much larger. This means our data is skewed, and that makes it hard for a neural network to handle. Let's fit 
our two features into a range of 0-1, by dividing the grades by 4.0, and the test score by 800.
'''
def scale_data(one_hot_data):
	# Making a copy of our data
	# Copying our data
	processed_data = one_hot_data[:]

	# Scaling the columns
	processed_data['gre'] = processed_data['gre']/800
	processed_data['gpa'] = processed_data['gpa']/4.0

	# Printing the first 10 rows of our procesed data
	print("processed_data")
	print(processed_data[:10])

	return processed_data

#################### Splitting the data into Training and Testing ####################
'''
In order to test our algorithm, we'll split the data into a Training and a Testing set. The size of the testing set will be 10% 
of the total data.
'''
def split_data(processed_data):
	sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
	train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

	print("Number of training samples is", len(train_data))
	print("Number of testing samples is", len(test_data))
	print(train_data[:10])
	print(test_data[:10])

	return train_data, test_data

#################### Splitting the data into features and targets (labels) ####################

# Now, as a final step before the training, we'll split the data into features (X) and targets (y).
# Also, in Keras, we need to one-hot encode the output. We'll do this with the to_categorical function.

def split_feature_and_target(train_data, test_data):
	# Separate data and one-hot encode the output
	# Note: We're also turning the data into numpy arrays, in order to train the model in Keras
	features = np.array(train_data.drop('admit', axis=1))
	targets = np.array(keras.utils.to_categorical(train_data['admit'], 2))
	features_test = np.array(test_data.drop('admit', axis=1))
	targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))
	
	print(features[:10])
	print(targets[:10])

	return features, targets, features_test, targets_test

def splitTrainTestData():
	#1. read the CSV
	data = readCSV()

	#2. Plotting the data		
	#plot_points(data)

	#3. data_rank
	#data_rank(data)
	
	#4. Onehot encoding
	one_hot_data = one_hot_coding(data)
	
	#5. Scale the data
	processed_data = scale_data(one_hot_data)

	#6. Split the data 
	train_data, test_data = split_data(processed_data)
	
	# In order to test our algorithm, we'll split the data into a Training and a Testing set. 
	# The size of the testing set will be 10% of the total data.
	#7. split feature and target
	features, targets, features_test, targets_test = split_feature_and_target(train_data, test_data)

	return features, targets, features_test, targets_test
