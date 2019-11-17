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
from hyperparams    import title, target_h, target_w

def get_model():
    cnn = Sequential()
    cnn.add(Conv2D(32, kernel_size=(3,3), input_shape=(target_w, target_h, 1), padding='same', activation='elu'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='elu'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='elu'))
    cnn.add(Dense(2,activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(cnn.summary())

    with open("model/arch/" + title + ".json", "w") as json_file:
        json_file.write(cnn.to_json())

    return cnn
