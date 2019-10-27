"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
This file loads the model saved from `otis.py` for testing &
evaluation.
"""
# Imports
from keras.models           import Sequential, model_from_json
from keras.utils            import to_categorical 
from keras.layers           import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing    import image

import matplotlib.pyplot as plt 
import numpy as np 

from otis import model, preprocess
from hyperparams import *

if __name__ == "__main__":
    # Find inputs
    (X_train, y_train, X_test, y_test) = preprocess(target_w, target_h) 

    # Reload model
    cnn = model(target_w, target_h)
    cnn.load_weights('model/model.h5')

    score = cnn.evaluate(X_test, y_test)
    print(score)