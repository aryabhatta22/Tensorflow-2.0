"""
Importing Libraries
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

print(tf.__version__)

"""
Data Preprocessing
"""

# classes in cifar10

class_name = [
                'airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
              ]

# Dataset import
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0

# image sample
plt.imshow(X_test[10])


"""
CNN Model
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import plot_model


cnn = Sequential()

# first layer
cnn.add( Conv2D( filters = 32, kernel_size = 3, padding = 'same', activation= 'relu', input_shape = [32,32,3] ) )

# second layer with Max Pooling
cnn.add( Conv2D( filters = 32, kernel_size = 3, padding = 'same', activation = 'relu' ) )
cnn.add( MaxPool2D( pool_size = 2, strides= 2, padding = 'valid' ))

# third layer
cnn.add( Conv2D( filters = 64, kernel_size = 3, padding = 'same', activation = 'relu' ) )

# fourth layer with Max Pooling
cnn.add( Conv2D( filters = 64, kernel_size = 3, padding = 'same', activation = 'relu' ) )
cnn.add( MaxPool2D( pool_size = 2, strides= 2, padding = 'valid' ))

# flattening layer
cnn.add( Flatten() )

# fully connected layer
cnn.add( Dense(units = 128, activation = 'relu') )

# output layer
cnn.add( Dense(units = 10, activation = 'softmax') )

# compile model
cnn.compile( 
            optimizer = 'adam', 
            loss = 'sparse_categorical_crossentropy', 
            metrics = ['sparse_categorical_accuracy'] 
            )

# model summary
cnn.summary()
plot_model(cnn, "Basic CNN Model.png", show_shapes = True)


# training model
cnn.fit( X_train, y_train, epochs = 10)

"""
Model Evaluation
"""

loss, acc = cnn.evaluate(X_test, y_test)
print("Test Accuracy: {}".format(acc))