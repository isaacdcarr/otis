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
#from __future__ import print_function

from keras.models   import Sequential
from keras.utils    import to_categorical 
from keras.layers   import Conv2D, MaxPool2D, Flatten, Dense
    # might not need to_categorical. Change output to classes rather than numbers.
from keras.preprocessing import image

import matplotlib as plt 
import numpy as np # might not need
import os

# Pre-processing
def preprocess(target_w, target_h): 
    X_train, y_train, X_test, y_test = [], [], [], []

    path = 'data/chest_xray/'
    train_path_normal       = path + 'train/NORMAL/'
    train_path_pneumonia    = path + 'train/PNEUMONIA/'
    test_path_normal        = path + 'test/NORMAL/'
    test_path_pneumonia        = path + 'test/PNEUMONIA/'
    
    print("... obtaining normal train data")
    i = 0 
    for file in os.listdir(train_path_normal):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_normal + file, target_size=(target_w, target_h))
            X_train.append(np.array(img))
            y_train.append(0)
    print("... ... Processed " + str(i) + " images")        
    
    print("... obtaining pneumonia train data")
    i = 0 
    for file in os.listdir(train_path_pneumonia):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_pneumonia + file, target_size=(target_w, target_h))
            X_train.append(np.array(img))
            y_train.append(1)
    print("... ... Processed " + str(i) + " images")     

    print("... obtaining normal test data")
    i = 0 
    for file in os.listdir(test_path_normal):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_normal + file, target_size=(target_w, target_h))
            X_test.append(np.array(img))
            y_test.append(0)
    print("... ... Processed " + str(i) + " images")     
    
    print("... obtaining pneumonia tests data")
    i = 0
    for file in os.listdir(test_path_pneumonia):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_pneumonia + file, target_size=(target_w, target_h))
            X_test.append(np.array(img))
            y_test.append(1)
    print("... ... Processed " + str(i) + " images")     

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    return (X_train, y_train, X_test, y_test)

def model(target_w, target_h):
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(5,5), input_shape=(target_w, target_h, 1), padding='same', activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Conv2D(6, kernel_size=(5,5), padding='same', activation='relu'))
    cnn.add(MaxPool2D())
    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dense(2,activation='softmax'))
    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(cnn.summary())
    return cnn


if __name__ == '__main__':
    os.system("clear")
    print("== Starting Otis ==")
    # hyperparameters
    epochs = 10
    target_w = 224
    target_h = 224

    # Run model 
    (X_train, y_train, X_test, y_test) = preprocess(target_w, target_h) 
    print("... define model")
    cnn = model(target_w, target_h) 
    print("... fit model")
    history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_train, y_train))

    # plot results
    plt.plot(history_cnn.history_cnn['acc'])
    plt.plot(history_cnn.history_cnn['val_acc'])

    # Evalute
    score = cnn.evaluate(X_test, y_test)
    print(score)
 