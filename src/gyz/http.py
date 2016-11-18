import httplib
import time
import sys
import os;
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
from loader import load
import numpy as np
import numpy
from keras import backend as K
K.set_image_dim_ordering('th')

model = load_model("model1.h5")

httpClient = None
try:
	while True:  
		httpClient = httplib.HTTPConnection('127.0.0.1', 8000, timeout = 30)
		httpClient.request('GET', '/')
		response = httpClient.getresponse()
		inp = map(int, response.read().replace('\r\n', '').split(' '))
		X_test = np.empty((1, 2, 37, 65), dtype = 'int32')
		for channel in range(2):
			for r in range(37):
				for c in range(65):
					X_test[0, channel, r, c] = int(inp[r * 2 * 65 + c * 2 + channel])

		(X_mean, X_scale) = (83.647755, 143.887664)
		(Y_mean, Y_scale) = (-6.418524, 93.565776)
		X_test = (X_test - X_mean) / X_scale
		Y_test = model.predict(X_test, batch_size = 1)
		Y_test = Y_test * Y_scale + Y_mean

		oup = "";
		for i in range(60):
			oup += str(Y_test[0][i]) + "?";
		httpClient = httplib.HTTPConnection('127.0.0.1', 8000, timeout = 30)
		httpClient.request('GET', '/' + oup);
		response = httpClient.getresponse()
except Exception, e:
	print e
finally:
	if httpClient:
		httpClient.close()
