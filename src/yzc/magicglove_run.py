from data_process import load_data
from keras.models import load_model

model = load_model('magicglove_model.h5')
path = '../../data/'
(X_train, Y_train), (X_test, Y_test) = load_data(path + '20161111/data.txt', 
validation_split = 0.2, start = 1, end = -1)

scale = 84.162565
mean = -6.687465
result = model.predict(X_train)
result = result * scale + mean