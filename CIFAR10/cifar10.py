#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:04:03 2019

@author: nithushan
"""

#Import all the libraries

import keras
from keras.datasets import cifar10
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization

from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras import regularizers

# Load the CIFAR10 Dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = np_utils.to_categorical(y_train,10)
y_test =  np_utils.to_categorical(y_test,10)

# Augment the data to create more images to train

data_augmentee = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
data_augmentee.fit(x_train)

# Fix the L2 Regularizers weights
poids= 0.0005


#SCreate the model architecture

model = Sequential()
model.add(Convolution2D(32, (3,3), padding='same',kernel_regularizer=regularizers.l2(poids), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(poids)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(poids)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile the model with Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
model.fit_generator(data_augmentee.flow(x_train, y_train, 1),steps_per_epoch = x_train.shape[0],epochs=3,verbose=1,validation_data=(x_test,y_test))

#model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=3)

# Save the model in h5 format
model.save('model_regu_l2.h5')


# Evaluate the accuracy of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))




