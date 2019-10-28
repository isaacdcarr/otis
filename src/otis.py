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
from keras.models           import Sequential, model_from_json
from keras.utils            import to_categorical 
from keras.layers           import Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU
from keras.preprocessing    import image

import matplotlib.pyplot as plt 
import numpy as np 
import os, json, csv 
import scipy.io as sio
import datetime as dt

from hyperparams import *

title = str(dt.datetime.now()) + ' ' + iter_num + 'iter_' + str(epochs) + 'epochs_' + str(target_w) + 'size'

# Pre-processing
def preprocess(target_w, target_h): 
    X_train, y_train, X_test, y_test, X_val, y_val = [], [], [], [], [], []

    path                    = 'data/chest_xray/'
    train_path_normal       = path + 'train/NORMAL/'
    train_path_pneumonia    = path + 'train/PNEUMONIA/'
    test_path_normal        = path + 'test/NORMAL/'
    test_path_pneumonia     = path + 'test/PNEUMONIA/'
    val_path_normal         = path + 'val/NORMAL/'
    val_path_pneumonia      = path + 'val/PNEUMONIA/'
    second_root_path        = 'data/rsna-dataset/'
    second_dataset_csv      = second_root_path + 'stage_2_train_labels.csv'
    second_dataset_path     = second_root_path + 'data/rsna-dataset/'

    print("... obtaining normal train data")
    num_img = 0 
    for file in os.listdir(train_path_normal):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(0)
    print("... ... Processed " + str(num_img) + " images")        
    
    print("... obtaining pneumonia train data")
    num_img = 0 
    for file in os.listdir(train_path_pneumonia):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(1)
    print("... ... Processed " + str(num_img) + " images")     

    print("... obtaining normal test data")
    num_img = 0 
    for file in os.listdir(test_path_normal):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(0)
    print("... ... Processed " + str(num_img) + " images")     
    
    print("... obtaining pneumonia tests data")
    num_img = 0
    for file in os.listdir(test_path_pneumonia):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(1)
    print("... ... Processed " + str(num_img) + " images")     

    print("... obtaining val train data")
    num_img = 0 
    for file in os.listdir(val_path_normal):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(val_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_val.append(np.array(img))
            y_val.append(0)
    print("... ... Processed " + str(num_img) + " images")        
    
    print("... obtaining pneumonia val data")
    num_img = 0 
    for file in os.listdir(val_path_pneumonia):
        num_img += 1
        print("... ... Processed " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(val_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_val.append(np.array(img))
            y_val.append(1)
    print("... ... Processed " + str(num_img) + " images")

    print("... obtaining second dataset ")
    with open(second_dataset_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        num_rows = 0
        num_fail = 0 
        for row in csv_reader:
            try: 
                img = image.load_img(second_dataset_path + row["patientId"] + '.png', target_size=(target_w, target_h), color_mode="grayscale")
                X_train.append(np.array(img))
                y_train.append(int(row["target"]))
                #print("... ... Processed " + str(num_rows) + " images", end="\r")
                num_rows += 1
            except: 
                num_fail += 1 
            print("... ... Processed:\t" + str(num_rows) + ", failed:\t" + str(num_fail), end="\r")
    print("... ... Processed " + str(num_rows) + " images")
    print("... ... Could not load " + str(num_fail) + " images")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)
    X_val   = np.array(X_val)
    y_val   = np.array(y_val)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test  = X_test.reshape(X_test.shape + (1,))
    X_val   = X_val.reshape(X_val.shape + (1,))
    y_train = to_categorical(y_train, 2)
    y_test  = to_categorical(y_test, 2)
    y_val   = to_categorical(y_val, 2)

    # print("X_train:\t" + str(X_train.shape)) 
    # print("y_train:\t" + str(y_train.shape)) 
    # print("X_test:\t" + str(X_test.shape)) 
    # print("y_test:\t" + str(y_test.shape)) 

    return (X_train, y_train, X_test, y_test, X_val, y_val)

def model(target_w, target_h):
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3,3), input_shape=(target_w, target_h, 1), padding='same', activation='elu'))
    cnn.add(MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='elu'))
    cnn.add(MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='elu'))
    cnn.add(Dense(2,activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(cnn.summary())

    with open("model/model_architecture" + title + ".json", "w") as json_file:
        json_file.write(cnn.to_json())

    return cnn

def main(): 
    print("== Starting Otis ==")
    print(title)

    print("... hyperparams")
    print("...    epochs:\t" + str(epochs))
    print("...    targ_w:\t" + str(target_w))
    print("...    targ_h:\t" + str(target_h))

    # Run model 
    (X_train, y_train, X_test, y_test, X_val, y_val) = preprocess(target_w, target_h) 
    print("... define model")
    cnn = model(target_w, target_h) 

    print(">>> For epochs: " + str(epochs))
    print("... fit model")
    history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test))
    cnn.save_weights("model/weights/weights_" + title  +'.model')
    cnn.save("model/model/entire_model_" + title + '.hdf5')

    # show gathered data
    try:
        json.dump(history_cnn.history, open('model/history/history_' + title, 'w'))
    except:
        json.dump(str(history_cnn.history), open('model/history/history_' + title, 'w'))
        print("Saved history as str")
        
    # Plot accuracy
    plt.plot(history_cnn.history['accuracy'])
    plt.plot(history_cnn.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('epoch')   
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model/img/acc_' + title + '.png')
    #plt.show()
    plt.clf()

    # Plot loss
    plt.plot(history_cnn.history['loss'])
    plt.plot(history_cnn.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model/img/loss_' + title + '.png')
    #plt.show()
    plt.clf()

    # Evalute
    score = cnn.evaluate(X_test, y_test)
    print(score)
    
    # Save data 
    sio.savemat('data/results/' + title + '.mat', history_cnn.history)
 
if __name__ == '__main__':
    main()
