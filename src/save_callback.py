"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file defines hyperparameters for the convolutional
neural network.
"""

import keras

from otis_results import get_results

class SaveResultsCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {} 
        self.epochs.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if len(self.epochs) % 5 == 0:
            get_results(self.history)