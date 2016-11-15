from data_process import load_data

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

batch_size = 128
nb_epoch = 100

img_rows, img_cols = 37, 65
img_channels = 2

path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161111/data.txt', 
validation_split = 0.2, start = 1, end = -1)

Y_train_origin = np.copy(Y_train)
Y_test_origin = np.copy(Y_test)

mean = np.mean(Y_train)
Y_train -= mean
scale = np.max(np.abs(Y_train)) 
Y_train /= scale


Y_test -= mean
Y_test /= scale

model = Sequential()

model.add(Convolution2D(8, 5, 5, border_mode = 'valid', input_shape = X_train.shape[1:]))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(16, 3, 3, border_mode = 'valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(256))
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

print ('mean = %f'%(mean))
print ('scale = %f'%(scale))

result = model.predict(X_train)
result = result * scale + mean
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
result = result * scale + mean
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