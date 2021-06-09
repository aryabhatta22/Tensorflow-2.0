"""
Importing Libraries
"""


import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import plot_model

"""
Data Preprocessing
"""

# Parameters
number_of_words = 20000
max_len = 100

# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

# padding sequence to same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen = max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen = max_len)


"""
RNN Model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

rnn = Sequential()

# embedding layer
rnn.add( 
        Embedding( 
                    input_dim = number_of_words, 
                    output_dim=128, 
                    input_shape = (X_train.shape[1],)
                    ) 
    )


# lstm layer
rnn.add( LSTM(units=128, activation='tanh') )

# output layer
rnn.add( Dense(units = 1, activation ='sigmoid') )

#compile model
rnn.compile( optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# model summary
rnn.summary()
plot_model(rnn, "Basic CNN Model.png", show_shapes = True)

# train model
rnn.fit( X_train, y_train, epochs = 3, batch_size=128 )

"""
Model Evaluation
"""

loss, acc = rnn.evaluate(X_test, y_test)
print("Test Accuracy: {}".format(acc))