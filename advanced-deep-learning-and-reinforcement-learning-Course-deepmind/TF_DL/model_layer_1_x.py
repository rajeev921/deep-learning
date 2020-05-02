# Import useful libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

class Model_1_x:
    def __init__(self, settings, log_period_samples, batch_size):
        # Store results of runs with different configurations in a dictionary.
        self.experiments_task1 = []
        self.experiments_task2 = []
        self.experiments_task3 = []
        self.experiments_task4 = []
        self.experiments_task5 = []
        self.settings = settings
        self.log_period_samples = log_period_samples
        self.batch_size = batch_size
        
    # Placeholders to feed train and test data into the graph.
    # Since batch dimension is 'None', we can reuse them both for train and eval.
    def get_placeholders(self):
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        return x, y_

    def evaluation_metrics(self):
        #self.plot_learning_curves([self.experiments_task4, self.experiments_task5])
        #self.plot_learning_curves([self.experiments_task1, self.experiments_task2, self.experiments_task3, self.experiments_task4, self.experiments_task5])
        self.plot_learning_curves([self.experiments_task1, self.experiments_task2, self.experiments_task3, self.experiments_task4, self.experiments_task5])
        #self.plot_summary_table([self.experiments_task1, self.experiments_task2, self.experiments_task3, self.experiments_task4])
        
    # Import dataset with one-hot encoding of the class labels.
    def get_data(self):
        return input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Plot learning curves of experiments
    def plot_learning_curves(self, experiment_data):
        # Generate figure.
        fig, axes = plt.subplots(3, 5, figsize=(22,12))
        st = fig.suptitle("Learning Curves for all Tasks and Hyper-parameter settings", fontsize="x-large")
        # Plot all learning curves.
        for i, results in enumerate(experiment_data):
            for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
                # Plot.
                xs = [x * self.log_period_samples for x in range(1, len(train_accuracy)+1)]
                axes[j, i].plot(xs, train_accuracy, label='train_accuracy')
                axes[j, i].plot(xs, test_accuracy, label='test_accuracy')
                # Prettify individual plots.
                axes[j, i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                axes[j, i].set_xlabel('Number of samples processed')
                axes[j, i].set_ylabel('Epochs: {}, Learning rate: {}.  Accuracy'.format(*setting))
                axes[j, i].set_title('Task {}'.format(i + 1))
                axes[j, i].legend()
        # Prettify overall figure.
        plt.tight_layout()  
        st.set_y(0.95)
        fig.subplots_adjust(top=0.91)
        plt.show()
  
    # Generate summary table of results.
    def plot_summary_table(self, experiment_data):
        # Fill Data.
        cell_text = []
        rows = []
        columns = ['Setting 1', 'Setting 2', 'Setting 3']
        for i, results in enumerate(experiment_data):
            rows.append('Model {}'.format(i + 1))
            cell_text.append([])
            for j, (setting, train_accuracy, test_accuracy) in enumerate(results):
                cell_text[i].append(test_accuracy[-1])
        
        # Generate Table.
        fig=plt.figure(frameon=False)
        ax = plt.gca()
        the_table = ax.table(
            cellText=cell_text,
            rowLabels=rows,
            colLabels=columns,
            loc='center')
        the_table.scale(1, 4)
        #Prettify
        ax.patch.set_facecolor('None')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
    '''
    Model 1 - Train a neural network model consisting of 1 linear layer, followed by a softmax:
    (input → linear layer → softmax → class probabilities)

    Hyper-parameters
    Train the model with three different hyper-parameter settings:

    num_epochs=5, learning_rate=0.0001
    num_epochs=5, learning_rate=0.005
    num_epochs=5, learning_rate=0.1
    '''
    # Train Model 1 with the different hyper-parameter settings.
    def ModelLayer_0(self):
        for (num_epochs, learning_rate) in self.settings:
            # Reset graph, recreate placeholders and dataset.
            tf.reset_default_graph()
            x, y_ = self.get_placeholders()
            mnist = self.get_data()
            eval_mnist = self.get_data()

            # Define model, loss, update and evaluation metric. 
            initializer = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(initializer([784, 10]))
            b = tf.Variable(initializer([10]))
            logits = tf.matmul(x,w)+b
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Train.
            i, train_accuracy, test_accuracy = 0, [], []
            log_period_updates = int(self.log_period_samples / self.batch_size)
            with tf.train.MonitoredSession() as sess:
                while mnist.train.epochs_completed < num_epochs:
                    #Update
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    #Training step
                    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

                    #Periodically evaluate
                    if i % log_period_updates == 0:
                        #Compute and store train accuracy on 20% training data.
                        a = 0.2
                        ex = eval_mnist.train.images
                        ey = eval_mnist.train.labels
                        size = int(ey.shape[0]*a)
                        part_ex = ex[0:size,:]
                        part_ey = ey[0:size,:]
                        train = sess.run(accuracy,feed_dict={x:part_ex,y_:part_ey})
                        print("%d th iter train accuracy %f" %(i, train))
                        train_accuracy.append(train)  

                        # Compute and store test accuracy. 
                        test = sess.run(accuracy,feed_dict={x:eval_mnist.test.images,y_:eval_mnist.test.labels})
                        print("%d th iter test accuracy %f" %(i,test))  
                        test_accuracy.append(test)   
            
            # save in a list
            self.experiments_task1.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))

    '''
    Model 2  - 1 hidden layer (32 units) with a ReLU non-linearity, followed by a softmax.
    (input  → non-linear layer  → linear layer  → softmax  → class probabilities)

    Hyper-parameters
    Train the model with three different hyper-parameter settings:

    num_epochs=15, learning_rate=0.0001
    num_epochs=15, learning_rate=0.005
    num_epochs=15, learning_rate=0.1

    '''
    def ModelLayer_1(self):
        self.settings = [(15, 0.0001), (15, 0.005), (15, 0.1)]
        for (num_epochs, learning_rate) in self.settings:
            tf.reset_default_graph()
            x, y_ = self.get_placeholders()
            mnist = self.get_data()
            eval_mnist = self.get_data()

            initializer = tf.contrib.layers.xavier_initializer()

            # non-linear layer 
            w_1 = tf.Variable(initializer([784, 32]))
            b_1 = tf.Variable(initializer([32]))
            h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

            #Linear layer
            w_2 = tf.Variable(initializer([32,10]))
            b_2 = tf.Variable(initializer([10]))
            logits = tf.matmul(h_1, w_2) + b_2
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            #evaluation
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

             # Train.
            i, train_accuracy, test_accuracy = 0, [], []
            log_period_updates = int(self.log_period_samples / self.batch_size)
            with tf.train.MonitoredSession() as sess:
                while mnist.train.epochs_completed < num_epochs:
                    # Update.
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Training step
                    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

                    # Periodically evaluate.
                    if i % log_period_updates == 0:

                        # Compute and store train accuracy on 20% training data.
                        a = 0.2
                        ex = eval_mnist.train.images
                        ey = eval_mnist.train.labels
                        size = int(ey.shape[0]*a)
                        part_ex = ex[0:size,:]    
                        part_ey = ey[0:size,:]
                        train = sess.run(accuracy,feed_dict={x:part_ex,y_:part_ey})
                        print("%d th iter train accuracy %f" %(i,train)) 
                        train_accuracy.append(train)  
                        
                        # Compute and store test accuracy. 
                        test = sess.run(accuracy,feed_dict={x:eval_mnist.test.images,y_:eval_mnist.test.labels})
                        print("%d th iter test accuracy %f" %(i,test))  
                        test_accuracy.append(test) 
            
                self.experiments_task2.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))  

    '''
    Model 3 - 2 hidden layers (32 units) each, with ReLU non-linearity, followed by a softmax.

    (input  →  non-linear layer  →  non-linear layer  →  linear layer  →  softmax  →  class probabilities)

    Hyper-parameters
    Train the model with three different hyper-parameter settings:

    num_epochs=5, learning_rate=0.003

    num_epochs=40, learning_rate=0.003

    num_epochs=40, learning_rate=0.05
    '''
    def ModelLayer_3(self):
        self.settings = [(5, 0.003), (40, 0.003), (40, 0.05)]
        for (num_epochs, learning_rate) in self.settings:
            tf.reset_default_graph()
            x, y_ = self.get_placeholders()
            mnist = self.get_data()
            eval_mnist = self.get_data()

            initializer = tf.contrib.layers.xavier_initializer()

            # non-linear layer 
            w_1 = tf.Variable(initializer([784, 32]))
            b_1 = tf.Variable(initializer([32]))
            h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

            # non-linear layer 
            w_2 = tf.Variable(initializer([32, 32]))
            b_2 = tf.Variable(initializer([32]))
            h_2 = tf.nn.relu(tf.matmul(h_1, w_2) + b_2)

            #Linear layer
            w_3 = tf.Variable(initializer([32,10]))
            b_3 = tf.Variable(initializer([10]))
            logits = tf.matmul(h_2, w_3) + b_3
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            #evaluation
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

             # Train.
            i, train_accuracy, test_accuracy = 0, [], []
            log_period_updates = int(self.log_period_samples / self.batch_size)
            with tf.train.MonitoredSession() as sess:
                while mnist.train.epochs_completed < num_epochs:
                    # Update.
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Training step
                    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

                    # Periodically evaluate.
                    if i % log_period_updates == 0:

                        # Compute and store train accuracy on 20% training data.
                        a = 0.2
                        ex = eval_mnist.train.images
                        ey = eval_mnist.train.labels
                        size = int(ey.shape[0]*a)
                        part_ex = ex[0:size,:]    
                        part_ey = ey[0:size,:]
                        train = sess.run(accuracy,feed_dict={x:part_ex,y_:part_ey})
                        print("%d th iter train accuracy %f" %(i,train)) 
                        train_accuracy.append(train)  
                        
                        # Compute and store test accuracy. 
                        test = sess.run(accuracy,feed_dict={x:eval_mnist.test.images,y_:eval_mnist.test.labels})
                        print("%d th iter test accuracy %f" %(i,test))  
                        test_accuracy.append(test) 
            
                self.experiments_task3.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))  

    '''
    Model 4 - 3 layer convolutional model (2 convolutional layers followed by max pooling) + 1 non-linear layer (32 units), followed by softmax.
    (input(28x28)  →  conv(3x3x8) + maxpool(2x2)  →  conv(3x3x8) + maxpool(2x2)  →  flatten  →  non-linear  →  linear layer  →  softmax  →  class probabilities)

    Use padding = 'SAME' for both the convolution and the max pooling layers.
    Employ plain convolution (no stride) and for max pooling operations use 2x2 sliding windows, with no overlapping pixels (note: this operation will down-sample the input image by 2x2).
    Use Stohastic Gradient Descent to train the models.
    
    Hyper-parameters
    Train the model with three different hyper-parameter settings:
    num_epochs=5, learning_rate=0.01
    num_epochs=10, learning_rate=0.001
    num_epochs=20, learning_rate=0.001
    '''
    def convModel(self):
        self.settings = [(5, 0.01), (10, 0.001), (20, 0.001)]

        for (num_epochs, learning_rate) in self.settings:
            tf.reset_default_graph()
            x, y_ = self.get_placeholders()
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            mnist = self.get_data()  # use for training.
            eval_mnist = self.get_data()  # use for evaluation.
            
            # Define model, loss, update and evaluation metric. 
            initializer = tf.contrib.layers.xavier_initializer()
            
            # conv layer 1
            w_conv1 = tf.Variable(initializer([3,3,1,8]))
            b_conv1 = tf.Variable(initializer([8]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # conv layer 2
            w_conv2 = tf.Variable(initializer([3,3,8,8]))
            b_conv2 = tf.Variable(initializer([8]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # flatten
            h_flat = tf.reshape(h_pool2, [-1, 7*7*8])
            
            # non-linear layer
            w_n = tf.Variable(initializer([7*7*8,32]))
            b_n = tf.Variable(initializer([32]))
            h_n = tf.nn.relu(tf.matmul(h_flat,w_n)+b_n)
            
            # linear layer + softmax & loss
            w_linear = tf.Variable(initializer([32,10]))
            b_linear = tf.Variable(initializer([10]))
            logits = tf.matmul(h_n, w_linear) + b_linear
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            
            # evalutaion
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Train.
            i, train_accuracy, test_accuracy = 0, [], []
            log_period_updates = int(self.log_period_samples / self.batch_size)
            with tf.train.MonitoredSession() as sess:
                while mnist.train.epochs_completed < num_epochs:
                    # Update.
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Training step 
                    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

                    # Periodically evaluate.
                    if i % log_period_updates == 0:        
                        # Compute and store train accuracy on 20% training data.
                        a=0.2
                        ex = eval_mnist.train.images
                        ey = eval_mnist.train.labels
                        size = int(ey.shape[0]*a)
                        part_ex = ex[0:size,:]    
                        part_ey = ey[0:size,:]
                        train = sess.run(accuracy,feed_dict={x:part_ex,y_:part_ey})
                        print("%d th iter train accuracy %f" %(i,train)) 
                        train_accuracy.append(train)  

                        # Compute and store test accuracy. 
                        test = sess.run(accuracy,feed_dict={x:eval_mnist.test.images,y_:eval_mnist.test.labels})
                        print("%d th iter test accuracy %f" %(i,test))  
                        test_accuracy.append(test)
                        
            self.experiments_task4.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))

    '''
    Model 5 - 3 layer convolutional model (2 convolutional layers followed by max pooling) + 1 non-linear layer (32 units), followed by softmax.
    (input(28x28)  →  conv(3x3x8) + maxpool(2x2)  →  conv(3x3x8) + maxpool(2x2)  →  flatten  →  non-linear  →  linear layer  →  softmax  →  class probabilities)

    Use padding = 'SAME' for both the convolution and the max pooling layers.
    Employ plain convolution (no stride) and for max pooling operations use 2x2 sliding windows, with no overlapping pixels 
    (note: this operation will down-sample the input image by 2x2).
    
    Replacing SGD with these optimizers(AdamOptimizer and RMSProp). 
    
    Hyper-parameters
    Train the model with three different hyper-parameter settings:
    num_epochs=5, learning_rate=0.01
    num_epochs=10, learning_rate=0.001
    num_epochs=20, learning_rate=0.001
    '''
    def convModelAdv(self):
        self.settings = [(5, 0.01), (10, 0.001), (20, 0.001)]

        for (num_epochs, learning_rate) in self.settings:
            tf.reset_default_graph()
            x, y_ = self.get_placeholders()
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            mnist = self.get_data()  # use for training.
            eval_mnist = self.get_data()  # use for evaluation.
            
            # Define model, loss, update and evaluation metric. 
            initializer = tf.contrib.layers.xavier_initializer()
            
            # conv layer 1
            w_conv1 = tf.Variable(initializer([3,3,1,8]))
            b_conv1 = tf.Variable(initializer([8]))
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # conv layer 2
            w_conv2 = tf.Variable(initializer([3,3,8,8]))
            b_conv2 = tf.Variable(initializer([8]))
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # flatten
            h_flat = tf.reshape(h_pool2, [-1, 7*7*8])
            
            # non-linear layer
            w_n = tf.Variable(initializer([7*7*8,32]))
            b_n = tf.Variable(initializer([32]))
            h_n = tf.nn.relu(tf.matmul(h_flat,w_n)+b_n)
            
            # linear layer + softmax & loss
            w_linear = tf.Variable(initializer([32,10]))
            b_linear = tf.Variable(initializer([10]))
            logits = tf.matmul(h_n, w_linear) + b_linear
            y = tf.nn.softmax(logits)
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits))
            #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            train_step = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss)

            
            # evalutaion
            correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Train.
            i, train_accuracy, test_accuracy = 0, [], []
            log_period_updates = int(self.log_period_samples / self.batch_size)
            with tf.train.MonitoredSession() as sess:
                while mnist.train.epochs_completed < num_epochs:
                    # Update.
                    i += 1
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)

                    # Training step 
                    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

                    # Periodically evaluate.
                    if i % log_period_updates == 0:        
                        # Compute and store train accuracy on 20% training data.
                        a=0.2
                        ex = eval_mnist.train.images
                        ey = eval_mnist.train.labels
                        size = int(ey.shape[0]*a)
                        part_ex = ex[0:size,:]    
                        part_ey = ey[0:size,:]
                        train = sess.run(accuracy,feed_dict={x:part_ex,y_:part_ey})
                        print("%d th iter train accuracy %f" %(i,train)) 
                        train_accuracy.append(train)  

                        # Compute and store test accuracy. 
                        test = sess.run(accuracy,feed_dict={x:eval_mnist.test.images,y_:eval_mnist.test.labels})
                        print("%d th iter test accuracy %f" %(i,test))  
                        test_accuracy.append(test)
                        
            self.experiments_task4.append(((num_epochs, learning_rate), train_accuracy, test_accuracy))
                        
                        
                        
                        
            
