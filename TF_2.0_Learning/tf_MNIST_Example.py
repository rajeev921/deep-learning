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

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts the model parameters to minimize the loss:
print("\nModel Fit \n")
model.fit(x_train, y_train, epochs=5)

#The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".
print("\nModel Evaluate \n")
model.evaluate(x_test,  y_test, verbose=2)

'''
The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.
If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
'''
probability_model = tf.keras.Sequential([
                                         model,
                                         tf.keras.layers.Softmax()
                                        ])

print("\n Probability_model \n")
print(probability_model(x_test[:5]))

