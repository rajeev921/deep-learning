import tensorflow as tf 
from tensorflow import keras
import numpy as np 

print(tf.__version__)

'''
Model 1 - Train a neural network model consisting of 1 linear layer, followed by a softmax:
(input → linear layer → softmax → class probabilities)

Hyper-parameters
Train the model with three different hyper-parameter settings:

num_epochs=5, learning_rate=0.0001
num_epochs=5, learning_rate=0.005
num_epochs=5, learning_rate=0.1
'''

class Model:
    def __init__(self, setting):
        # Store results of runs with different configurations in a dictionary.
        self.experiments_task1 = []
        self.settings = setting
        self.inp_mat = 784
        self.num_state = 10
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.get_data()

    # Placeholders to feed train and test data into the graph.
    # Since batch dimension is 'None', we can reuse them both for train and eval.
    def get_Variable(self):
        x = tf.Variable(tf.ones(shape=[None, self.inp_mat]), dtype=tf.float32)
        y_ = tf.Variable(tf.ones(shape=[None, num_state]), dtype=tf.float32)
        return x, y_

    # Import dataset with one-hot encoding of the class labels.
    def get_data(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        return train_images, train_labels, test_images, test_labels

    # Train Model 1 with the different hyper-parameter settings.
    def ModelLayer_1(self):
        for (num_epochs, learning_rate) in self.settings:
            # Reset graph, recreate placeholders and dataset.
            #tf.reset_default_graph()
            x, y_ = self.get_Variable()
            
            train_label = self.train_labels[:1000]
            test_label = self.test_labels[:1000]

            train_image = self.train_images[:1000].reshape(-1, 28 * 28) / 255.0
            test_image = self.test_images[:1000].reshape(-1, 28 * 28) / 255.0
            
            # Define model, loss, update and evaluation metric. 
            initializer = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(initializer([784,10]))
            b = tf.Variable(initializer([10]))
            logits = tf.matmul(x,w)+b
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        pass