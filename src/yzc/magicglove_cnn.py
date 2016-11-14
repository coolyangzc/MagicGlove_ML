from data_process import load_data

import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D

batch_size = 32
nb_epoch = 10

img_rows, img_cols = 37, 65
img_channels = 2

path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161111/data.txt', start = 1)

model = Sequential()

model.add(Convolution2D(4, 5, 5, border_mode = 'valid', input_shape = (2, 37, 65)))
model.add(Activation('tanh'))

model.add(Convolution2D(8, 3, 3, border_mode = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(60))
model.add(Activation('tanh'))

model.compile(loss = 'mean_squared_error', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = 100, nb_epoch = 50, shuffle = True, verbose = 1, show_accuracy = True, validation_split = 0.2)

result = model.predict(X[0 : 1], batch_size = 1)[0]
print result
print Y[0]
print result - Y[0]
dist = np.sqrt(np.sum(np.square(result - Y[0])))
print dist

