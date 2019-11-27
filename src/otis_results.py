"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file, given the results from the cnn, plots the accuracy and loss in MATLAB
and saves the plot. It then saves the dictionary as a `.mat` and a `.json` file 
so in case this data needs to be used at a later date.
"""
import json
import matplotlib.pyplot as plt 
import scipy.io as sio
from hyperparams import title
from keras import backend as K

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def get_results(history): 
    # Plot accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('epoch')   
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('results/img/acc_' + title + '.png')
    plt.clf()

    # Plot loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('results/img/loss_' + title + '.png')
    plt.clf()

    # Save data as `.mat`
    sio.savemat('results/history/mat/' + title + '.mat', history)

    # Save the data as json
    save_path = 'results/history/json/'
    try:
        json.dump(history, open(save_path + title + '.json', 'w'))
    except:
        json.dump(str(history), open(save_path + 'str' + title + '.json', 'w'))
        

