import os
import tensorflow as tf 
import numpy as np 

#define the graph
a = tf.constant(10)
b = tf.constant(20)
c = tf.multiply(a, b)
d = tf.add(a, b)
res = tf.divide(c, d)

# e = tf.zeros((int(1e12), int(1e12))) # perfectly defined quantity
# f = np.zeros((int(1e12), int(1e12))) # out of memory error

# Tensor shapes can usually be derived from the graph (shape inference).
tz = tf.zeros((10, 10)) # tz.shape → (10, 10)
print(tz.shape)
# tc = tf.concat([e, e], axis=0) # b.shape → (20, 10)

# Variables enable learning, by preserving state across execution of the graph
# All trainable parameters of machine learning models are tf.Variables.
v = tf.get_variable("name", dtype=tf.float32, shape=[2, 2], initializer=tf.random_normal_initializer(stddev=0.5))

y = tf.linalg.matmul(v, tf.constant([[1., 2.], [3., 4.]]))

# Variables can be assigned new values, and will maintain them across graph executions until the next update.
increment_op = v.assign(v + 1)
increment_op = tf.assign(v, v+1)


# Placeholders and feed
g = tf.placeholder(tf.float32, [])
h = tf.constant(1.0)
i = g + h

init = tf.global_variables_initializer()
#execute the graph
with tf.Session() as session:
	# session.run(init)
	print(session.run(d))
	print(session.run(res))
	print(session.run(i, feed_dict={g : 3.0}))
	