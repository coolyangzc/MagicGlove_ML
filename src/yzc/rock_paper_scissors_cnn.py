from data_process import load_data

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 128
nb_epoch = 30
X_scaling = True

img_rows, img_cols = 37, 65
img_channels = 2
 
path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161118/data.txt', 
validation_split = 0.2, start = 1, end = -1, mission = 'rock_paper_scissors', shuffle = True)

if (X_scaling):
    X_mean = np.mean(X_train)
    X_train -= X_mean
    X_scale = np.std(X_train)
    X_train /= X_scale
else:
    X_mean = 0.
    X_scale = 1.
X_test = (X_test - X_mean) / X_scale

print ('(X_mean, X_scale) = (%f, %f)'%(X_mean, X_scale))

model = Sequential()

model.add(Convolution2D(4, 5, 5, border_mode = 'valid', input_shape = X_train.shape[1:]))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(8, 3, 3, border_mode = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size = batch_size, 
          nb_epoch = nb_epoch,
          validation_data = (X_test, Y_test),
          shuffle = True)

#Validation

print ('(X_mean, X_scale) = (%f, %f)'%(X_mean, X_scale))

model.save('rock_paper_scissors_model.h5')