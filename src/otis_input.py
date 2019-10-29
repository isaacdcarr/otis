"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file gathers the data 

This file includes training, testing & validating.
"""

import os
import csv
import numpy as np 

from keras.preprocessing    import image
from keras.utils            import to_categorical 

from hyperparams import target_h, target_w

def get_input(): 
    X_train, y_train, X_test, y_test = [], [], [], []

    path                    = 'data/chest_xray/'
    train_path_normal       = path + 'train/NORMAL/'
    train_path_pneumonia    = path + 'train/PNEUMONIA/'
    test_path_normal        = path + 'test/NORMAL/'
    test_path_pneumonia     = path + 'test/PNEUMONIA/'
    val_path_normal         = path + 'val/NORMAL/'
    val_path_pneumonia      = path + 'val/PNEUMONIA/'

    second_root_path        = 'data/rsna-dataset/'
    second_dataset_csv      = second_root_path + 'stage_2_train_labels.csv'
    second_dataset_path     = second_root_path + 'train_img/'

    print("... obtaining first dataset ")
    num_img = 0 
    for file in os.listdir(train_path_normal):
        num_img += 1
        print("... ... Processed train norm " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(0)
    print("... ... Processed train norm " + str(num_img) + " images")        
    
    num_img = 0 
    for file in os.listdir(train_path_pneumonia):
        num_img += 1
        print("... ... Processed train pneu " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(train_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_train.append(np.array(img))
            y_train.append(1)
    print("... ... Processed train pneu " + str(num_img) + " images")     

    num_img = 0 
    for file in os.listdir(test_path_normal):
        num_img += 1
        print("... ... Processed test norm " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(0)
    print("... ... Processed test norm " + str(num_img) + " images")     
    
    num_img = 0
    for file in os.listdir(test_path_pneumonia):
        num_img += 1
        print("... ... Processed test pneu " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(test_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(1)
    print("... ... Processed test pneu " + str(num_img) + " images")     

    num_img = 0 
    for file in os.listdir(val_path_normal):
        num_img += 1
        print("... ... Processed val norm " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(val_path_normal + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(0)
    print("... ... Processed val norm " + str(num_img) + " images")        
    
    print("... obtaining pneumonia val data")
    num_img = 0 
    for file in os.listdir(val_path_pneumonia):
        num_img += 1
        print("... ... Processed val pneu " + str(num_img) + " images", end="\r")
        if 'jpeg' in file:
            img = image.load_img(val_path_pneumonia + file, target_size=(target_w, target_h), color_mode="grayscale")
            X_test.append(np.array(img))
            y_test.append(1)
    print("... ... Processed val pneum " + str(num_img) + " images")

    # print("... obtaining second dataset ")
    # with open(second_dataset_csv, mode='r') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     num_rows = 0
    #     num_fail = 0 
    #     for row in csv_reader:
    #         try: 
    #             img = image.load_img(second_dataset_path + row["patientId"] + '.png', target_size=(target_w, target_h), color_mode="grayscale")
    #             X_train.append(np.array(img))
    #             y_train.append(int(row["Target"]))
    #             num_rows += 1
    #         except Exception as e: 
    #             print(e)
    #             num_fail += 1 
    #         print("... ... Processed:\t" + str(num_rows) + ", failed:\t" + str(num_fail), end="\r")
    # print("... ... Processed:\t" + str(num_rows) + ", failed:\t" + str(num_fail))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    X_train = X_train.reshape(X_train.shape + (1,))
    X_test  = X_test.reshape(X_test.shape + (1,))
    y_train = to_categorical(y_train, 2)
    y_test  = to_categorical(y_test, 2)

    return (X_train, y_train, X_test, y_test)