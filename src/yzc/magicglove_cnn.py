from data_process import load_data

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 128
nb_epoch = 128
X_scaling = True
Y_scaling = True

img_rows, img_cols = 37, 65
img_channels = 2

path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161117/data-all.txt', 
validation_split = 0.2, start = 1, end = -2, shuffle = True)

if (X_scaling):
    X_mean = np.mean(X_train)
    X_train -= X_mean
    X_scale = np.std(X_train)
    X_train /= X_scale
else:
    X_mean = 0.
    X_scale = 1.
X_test = (X_test - X_mean) / X_scale

Y_train_origin = np.copy(Y_train)
Y_test_origin = np.copy(Y_test)

if (Y_scaling):
    Y_mean = np.mean(Y_train)
    Y_train -= Y_mean
    Y_scale = np.max(np.abs(Y_train))
    Y_train /= Y_scale
else:
    Y_mean = 0.
    Y_scale = 1.


Y_test = (Y_test - Y_mean) / Y_scale
print ('(X_mean, X_scale) = (%f, %f)'%(X_mean, X_scale))
print ('(Y_mean, Y_scale) = (%f, %f)'%(Y_mean, Y_scale))

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
model.add(Dense(60))
model.add(Activation('tanh'))

model.compile(loss = 'mse', optimizer = 'RMSprop')

model.fit(X_train, Y_train, 
          batch_size = batch_size, 
          nb_epoch = nb_epoch,
          validation_data = (X_test, Y_test),
          shuffle = True)

#Validation

print ('(X_mean, X_scale) = (%f, %f)'%(X_mean, X_scale))
print ('(Y_mean, Y_scale) = (%f, %f)'%(Y_mean, Y_scale))

result = model.predict(X_train)
result = result * Y_scale + Y_mean
dist = 0.
maxdist = 0.
for i in range(len(result)):
    max_tmp = 0.
    for j in range(0, 60, 3): 
        dist_tmp = np.sqrt(np.sum(np.square(result[i][j:j+3] - Y_train_origin[i][j:j+3])))
        dist += dist_tmp
        max_tmp = max(max_tmp, dist_tmp)
    maxdist += max_tmp
dist /= len(result) * 20
maxdist /= len(result)
print ('train dist = %f(avg), %f(max)'%(dist, maxdist))

result = model.predict(X_test)
result = result * Y_scale + Y_mean
dist = 0.
maxdist = 0.
for i in range(len(result)):
    max_tmp = 0.
    for j in range(0, 60, 3): 
        dist_tmp = np.sqrt(np.sum(np.square(result[i][j:j+3] - Y_test_origin[i][j:j+3])))
        dist += dist_tmp
        max_tmp = max(max_tmp, dist_tmp)
    maxdist += max_tmp
dist /= len(result) * 20
maxdist /= len(result)
print ('test dist = %f(avg), %f(max)'%(dist, maxdist))

model.save('magicglove_model.h5')