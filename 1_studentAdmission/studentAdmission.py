'''
Predicting Student Admissions with Neural Networks. The goal here is to predict 
if a student will be admitted to a graduate program based on these features. 

The dataset originally came from here: http://www.ats.ucla.edu/

This dataset has three input features: GRE score, GPA, and the rank of the 
undergraduate school (numbered 1 through 4). Institutions with rank 1 have the 
highest prestige, those with rank 4 have the lowest

'''

import PrepareData as prepD

# 8. prepare the data
features, targets, features_test, targets_test = prepD.splitTrainTestData()

# Training the 2-layer Neural Network¶
# The following function trains the 2-layer neural network. First, we'll write some helper functions.

# Activation (sigmoid) function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):
	return - y*np.log(output) - (1 - y) * np.log(1-output)

# TODO: Backpropagate the error¶
# Now it's your turn to shine. Write the error term. Remember that this is given by the equation −(𝑦−𝑦̂ )𝜎′(𝑥)
 
# TODO: Write the error term formula
def error_term_formula(y, output):
	return (y-output) * output * (1 - output)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

# Training function
def train_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here 
            #   rather than storing h as a separate variable 
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            #   Notice we calulate f'(h) here instead of defining a separate
            #   sigmoid_prime function. This just makes it faster because we
            #   can re-use the result of the sigmoid function stored in
            #   the output variable
            error_term = error_term_formula(y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights
    
weights = train_nn(features, targets, epochs, learnrate)


# Calculating the Accuracy on the Test Data
# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))