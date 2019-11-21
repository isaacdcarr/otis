from hyperparams import * 
from otis_model     import get_model
from keras.utils import plot_model
plot_model(get_model(), show_shapes=True, to_file='model.png')
