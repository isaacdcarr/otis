"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
This file aims to construct a complete convolutional neural network (CNN) from scratch.
It aims to be a proof of concept to see if a CNN can accurately detect pneumonia from a
chest x-ray.

This file includes training, testing & validating.
"""

# Imports
# from keras.models           import Sequential, model_from_json
# from keras.utils            import to_categorical 
# from keras.layers           import Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU
# from keras.preprocessing    import image

# import matplotlib.pyplot as plt 
# import numpy as np 
# import os, json, csv 
# import json
# import scipy.io as sio

from hyperparams    import *
from otis_input     import get_input
from otis_model     import get_model
from otis_results   import get_results

def main(): 
    print("== Starting Otis ==")
    print(title)
    print("... hyperparams")
    print("...    epochs:\t" + str(epochs))
    print("...    targ_w:\t" + str(target_w))
    print("...    targ_h:\t" + str(target_h))

    # Obtain input 
    (X_train, y_train, X_test, y_test, X_val, y_val) = get_input()
    
    # Define model
    print("... define model")
    cnn = get_model() 

    # Train the model
    print(">>> For epochs: " + str(epochs))
    print("... fit model")
    history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    
    # Locally save the weights and entire model 
    cnn.save_weights("model/weights/weights_" + title  +'.model')
    cnn.save("model/model/entire_model_" + title + '.hdf5')
    
    # Evalute
    score = cnn.evaluate(X_test, y_test)
    print(score)

    # Save results
    get_results(history_cnn)
 
if __name__ == '__main__':
    main()
