"""
Importing Libraries
"""

import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

"""
Data Preprocessing
"""

# Dataset import
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the data
# since ann has fully connected layer we flatten the image by reshaping it
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print('Reshaped training set dimesnion: ', X_train.shape)
print('Reshaped test set dimesnion: ', X_test.shape)

"""
ANN Model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import plot_model


# Define model
ann = Sequential()

# First layer
ann.add( Dense(units = 128, activation = 'relu', input_shape = (784, )) )
# Dropout
ann.add( Dropout(0.2) )
# Output Layer
ann.add( Dense(units = 10, activation = 'softmax') )

# Compile model
ann.compile( 
            optimizer = 'adam', 
            loss = 'sparse_categorical_crossentropy', 
            metrics = ['sparse_categorical_accuracy'] 
            )

# Model Summary
ann.summary()
plot_model(ann, "Basic ANN Model.png", show_shapes = True)

# Train model
ann.fit( X_train, y_train, epochs = 10)

"""
Model Evaluation
"""

loss, acc = ann.evaluate(X_test, y_test)
print("Test Accuracy: {}".format(acc))