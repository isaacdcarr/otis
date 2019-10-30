"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file defines hyperparameters for the convolutional
neural network.
"""

import datetime as dt

save_model      = False
single_input    = True
iter_num        = '4'
iter_tag        = 'data_split0.2'
iteration       = iter_num + '_' + iter_tag
epochs          = 100
target_w        = 224
target_h        = target_w
title           = dt.datetime.now().strftime("%Y-%m-%d_%H:%M") + '_' + iteration + '_' + str(epochs) + 'epochs_' + str(target_w) + 'size'
