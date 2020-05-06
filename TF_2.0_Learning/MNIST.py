import tensorflow as tf 

mnist = tf.keras.datasets.mnist 

# Convert int into float (Scale the value)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0 ,  x_test/255.0

# Choose an optimizer and loss function for training
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(x_train[:1]).numpy()

print(predictions)

# The tf.nn.softmax function converts these logits to "probabilities" for each class:
print(tf.nn.softmax(predictions))

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits 
# and a True index and returns a scalar loss for each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("Loss Function")
print(loss_fn)



