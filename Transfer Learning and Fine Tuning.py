"""
Importing Libraries
"""

import os
import zipfile     # to be used if the file is in zip format
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
Extract data
"""

# uncomment if file is in zip
'''
dataset_path = "./cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("./")
zip_object.close()
'''

dataset_path_new = "./cats_and_dogs_filtered/"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

"""
Model
"""

IMG_SHAPE = (128, 128, 3)

# Loading pre trained model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights="imagenet")

base_model.summary()

# freeze the base model from training
base_model.trainable = False

# Defining a custom head
print("current output dimesnion of base_model ", base_model.output)

global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
print("dimension after gloabl avg pooling layer ", global_avg_layer)

prediction_layer = tf.keras.layers.Dense( units=1, activation='sigmoid' ) (global_avg_layer)

#defining model
model = tf.keras.models.Model( inputs = base_model.input, outputs = prediction_layer)

model.summary()

# compiling model

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

"""
Data generator
"""

# generating data for mobilenet architecture of size 128 x 128

data_gen_train = ImageDataGenerator( rescale=1/255.)        # it also has other parameters which can be used
data_gen_valid = ImageDataGenerator( rescale=1/255.)

train_generator = data_gen_train.flow_from_directory( train_dir, target_size = (128,128), batch_size = 128, class_mode='binary')
valid_generator = data_gen_valid.flow_from_directory( validation_dir , target_size = (128,128), batch_size = 128, class_mode='binary')

"""
Model training
"""

model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

"""
Model Evaluation
"""

loss, acc = model.evaluate(valid_generator)
print("Test Accuracy: {}".format(acc))

"""
Fine Tuning

>> We unfreeze some layers of base model (!!!! dont unfreeze all layers) mostly top layer and train the data on them
"""

base_model.trainable = True
print("Number of layer in the base model: ", len(base_model.layers))

FINE_TUNE_AT = 100   # layer from where we are going to train

#freeze layer till 100
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# compiling fine tunes model
model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#training model
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

loss, acc = model.evaluate(valid_generator)
print("Test Accuracy after fine tuning: {}".format(acc))