"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file defines hyperparameters for the convolutional
neural network.
"""

# import datetime as dt

iter_num    = '3'
iter_tag    = 'newdata'
iteration   = iter_num + '_' + iter_tag
epochs      = 4
target_w    = 224
target_h    = target_w
title       = str(dt.datetime.now()) + '_' + iteration + '_' + str(epochs) + 'epochs_' + str(target_w) + 'size'