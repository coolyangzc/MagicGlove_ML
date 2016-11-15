from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from loader import load
from six.moves import range
import numpy
from keras import backend as K
K.set_image_dim_ordering('th')

inp, oup = load('../../data/20161111/data.txt', 100, 2700)
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
model.fit(inp, oup, batch_size = 100, nb_epoch = 50, shuffle = True, verbose = 1, show_accuracy = True, validation_split = 0.2)

result = model.predict(inp[0 : 1], batch_size = 1)[0]
print result
print oup[0]
print result - oup[0]
dist = numpy.sqrt(numpy.sum(numpy.square(result - oup[0])))
print dist
model.save('model.h5')
