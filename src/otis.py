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

from keras.models   import Sequential, model_from_json
from keras.utils    import to_categorical 
from keras.layers   import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing import image

import matplotlib.pyplot as plt 
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
            img = image.load_img(train_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(0)
    print("... ... Processed " + str(i) + " images")        
    
    print("... obtaining pneumonia train data")
    i = 0 
    for file in os.listdir(train_path_pneumonia):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(1)
    print("... ... Processed " + str(i) + " images")     

    print("... obtaining normal test data")
    i = 0 
    for file in os.listdir(test_path_normal):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(0)
    print("... ... Processed " + str(i) + " images")     
    
    print("... obtaining pneumonia tests data")
    i = 0
    for file in os.listdir(test_path_pneumonia):
        i += 1
        print("... ... Processed " + str(i) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(1)
    print("... ... Processed " + str(i) + " images")     

    plt.imshow(X_train[0],cmap='gray')

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test  = X_test.reshape(X_test.shape + (1,))
    y_train = to_categorical(y_train, 2)
    y_test  = to_categorical(y_test, 2)

    print("X_train:\t" + str(X_train.shape)) 
    print("y_train:\t" + str(y_train.shape)) 
    print("X_test:\t" + str(X_test.shape)) 
    print("y_test:\t" + str(y_test.shape)) 

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

    with open("model/model_architecture.json", "w") as json_file:
        json_file.write(cnn.to_json())

    return cnn


if __name__ == '__main__':
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
    cnn.save_weights("model/model.h5")

    # plot results
    plt.plot(history_cnn.history_cnn['acc'])
    plt.plot(history_cnn.history_cnn['val_acc'])
    plt.show()

    # Evalute
    score = cnn.evaluate(X_test, y_test)
    print(score)
 