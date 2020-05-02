
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

class PrepareData(object):
	#1. Load and prepare the data
	def __init__(self):
		print("init")

		print("Load the Dataset")
		self.ride_data = self.loadData()		
		
		print("plot both the data")
		self.plot_data()

		print("categorical value for both data")
		self.data = self.dummy_variable()
		#days_data  = self.dummy_variable(days_ride)		

		print("Scale the target variable")
		self.scaled_features = {}
		self.scale_target()

		print("Split the dataset into training and testing ")
		self.test_data, features, targets, self.test_features, self.test_targets =  self.split_dataset()

		# Hold out the last 60 days or so of the remaining data as a validation set
		self.train_features, self.train_targets = features[:-60*24], targets[:-60*24]
		self.val_features, self.val_targets = features[-60*24:], targets[-60*24:]

		print("Finishing data preparation. Now we have training validation and testing data")

		
	def getData(self):
		return self.train_features, self.train_targets, self.val_features, self.val_targets, self.test_features, self.test_targets

	def loadData(self):
		data_path = "Bike-Sharing-Dataset/hour.csv"
		#data_path  = "Bike-Sharing-Dataset/day.csv"
		ride = pd.read_csv(data_path) 
		print("hours_ride head\n", ride.head())
		return ride 

	#2. Plot the data
	def plot_data(self):
		self.ride_data[:24*10].plot(x='dteday', y='cnt')
		plt.show

	#3. Dummy variables
	'''
	Here we have some categorical variables like season, weather, month. To include these 
	in our model, we'll need to make binary dummy variables. This is simple to do with 
	Pandas thanks to get_dummies()
	'''
	def dummy_variable(self):
		dummy_fields = ['season', 'mnth', 'hr', 'weekday']
		for each in dummy_fields:
			dummies = pd.get_dummies(self.ride_data[each], prefix=each, drop_first=False)
			self.ride_data = pd.concat([self.ride_data, dummies], axis=1)
		
		fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday', 'atemp', 'mnth', 'workingday', 'hr']
		data = self.ride_data.drop(fields_to_drop, axis=1)
		print(data.head())

		return data
	
	#4. Scaling target variables
	'''
	To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and 
	scale the variables such that they have zero mean and a standard deviation of 1.
	The scaling factors are saved so we can go backwards when we use the network for predictions.	
	'''
	def scale_target(self):
		quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
		# Store scalings in a dictionary so we can convert back later
		
		for each in quant_features:
			mean, std = self.data[each].mean(), self.data[each].std()
			self.scaled_features[each] = [mean, std]
			self.data.loc[:, each] = (self.data[each] - mean)/std

		print(self.data.head())



	#5. Splitting the data into training, testing, and validation sets
	'''
	We'll save the data for the last approximately 21 days to use as a test set after we've trained the network. 
	We'll use this set to make predictions and compare them with the actual number of riders.
	'''
	def split_dataset(self):
		#Save data for approximately the last 21 days 
		test_data = self.data[-21*24:]

		# Now remove the test data from the data set 
		self.data = self.data[:-21*24]

		# Separate the data into features and targets
		target_fields = ['cnt', 'casual', 'registered']
		features, targets = self.data.drop(target_fields, axis=1), self.data[target_fields]
		test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

		return test_data, features, targets, test_features, test_targets

	# Check out your predictions
	'''
	Here, use the test data to view how well your network is modeling the data. If something is completely wrong 
	here, make sure each step in your network is implemented correctly.
	'''
	def plotPredictions(self, network):
		fig, ax = plt.subplots(figsize=(8,4))

		mean, std = self.scaled_features['cnt']
		predictions = network.run(self.test_features).T*std + mean
		ax.plot(predictions[0], label='Prediction')
		ax.plot((self.test_targets['cnt']*std + mean).values, label='Data')
		ax.set_xlim(right=len(predictions))
		ax.legend()

		dates = pd.to_datetime(self.ride_data.iloc[self.test_data.index]['dteday'])
		dates = dates.apply(lambda d: d.strftime('%b %d'))
		ax.set_xticks(np.arange(len(dates))[12::24])
		_ = ax.set_xticklabels(dates[12::24], rotation=45)
