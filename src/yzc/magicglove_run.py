from data_process import load_data
from keras.models import load_model

model = load_model('magicglove_model_3.11_9.09.h5')
path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161111/data.txt', 
shuffle = False, validation_split = 0.2, start = 1, end = -1)

X_mean = 83.619129
X_scale = 143.878194
Y_mean = -6.413624
Y_scale = 93.570676
X_train = (X_train - X_mean) / X_scale
result = model.predict(X_train)
result = result * Y_scale + Y_mean