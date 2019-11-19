"""
Isaac Carr (i.carr@unsw.edu.au)
Developed for MMAN4020, 19T3
Health Group 4
---
This file constructs the CNN using keras.
"""

# import json
from keras.models   import Sequential
from keras.layers   import Conv2D, MaxPool2D, Flatten, Dense, LeakyReLU
from keras.layers.normalization import BatchNormalization
from hyperparams    import title, target_h, target_w

def get_model():
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3,3), input_shape=(target_w, target_h, 1), kernel_initializer='glorot_uniform'))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())

    cnn.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Conv2D(64, kernel_size=(3,3)))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())

    cnn.add(MaxPool2D(pool_size=(2,2),stides=(2,2)))
    
    cnn.add(Flatten())

    cnn.add(Dense(1024))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())

    cnn.add(Dense(256))
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(BatchNormalization())

    cnn.add(Dense(1,activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(cnn.summary())

    with open("model/arch/" + title + ".json", "w") as json_file:
        json_file.write(cnn.to_json())

    return cnn
