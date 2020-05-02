import os
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as mlt 

# Data
num_samples, w, b = 20, 0.5, 2.
xs = np.asarray(range(num_samples))
ys = np.asarray([x * w + b + np.random.normal() for x in range(num_samples)])

mlt.plot(xs, ys)
mlt.show()

# Linear regression - Model
# The linear model is parametrized by two variables only: the slope (w) and the offset (b).
class Linear(object):
	def __init__(self):
		self.w = tf.compat.v1.get_variable("w", dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())
		self.b = tf.compat.v1.get_variable("b", dtype=tf.float32, shape=[], initializer=tf.zeros_initializer())

	def __call__(self, x):
		return self.w * x + self.b

# We can define the solver for the linear regression problem as part of the graph itself.
#Solver
xtf = tf.compat.v1.placeholder(tf.float32, [num_samples], "xs")
ytf = tf.compat.v1.placeholder(tf.float32, [num_samples], "ys")
model = Linear()
model_output = model(xtf)

cov = tf.reduce_sum(xtf - tf.reduce_sum(xtf)) * (ytf - tf.reduce_sum(ytf))
var = tf.reduce_sum(tf.square(xtf - tf.reduce_sum(xtf)))

w_hat = cov / var  
b_hat = tf.reduce_sum(ytf) - w_hat * tf.reduce_sum(xtf)

solve_w = model.w.assign(w_hat)
solve_b = model.w.assign(tf.reduce_sum(ytf) - w_hat * tf.reduce_sum(xtf))

# Execution
with tf.train.MonitoredSession() as sess:
	sess.run(
		[solve_w, solve_b],
		feed_dict={xtf: xs, ytf:ys})
	'''
	preds = sess.run(
				model_output,
				feed_dict={xtf: xs, ytf:ys})
	'''