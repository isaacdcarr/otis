"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
The file defines hyperparameters for the convolutional
neural network.
"""

import datetime as dt

save_model      = True
single_input    = False
iter_num        = '18'
val_split       = 0.1
iter_tag        = 'singledata_quickresults'
iteration       = iter_num + '_' + iter_tag
epochs          = 30
target_w        = 250
target_h        = target_w
title           = dt.datetime.now().strftime("%Y-%m-%d_%H:%M") + \
                    '_' + iteration + '_' + str(epochs) + \
                    'epochs_' + str(target_w) + 'size'
