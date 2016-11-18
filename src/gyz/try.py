from keras.models import load_model

from keras import backend as K
K.set_image_dim_ordering('th')

model = load_model('model1.h5')

