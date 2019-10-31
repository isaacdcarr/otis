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
from keras.callbacks import CSVLogger

from hyperparams    import *
from otis_input     import get_input_train_and_test, get_single_input
from otis_model     import get_model
from otis_results   import get_results
from save_callback  import SaveResultsCallback

def main(): 
    print("== Starting Otis ==")
    print(title)
    print("... hyperparams")
    print("...    epochs:\t" + str(epochs))
    print("...    targ_w:\t" + str(target_w))
    print("...    targ_h:\t" + str(target_h))

    # Define model
    cnn = get_model() 

    # Define graphing callback & csvlogger callback
    csv_path = 'results/history/csv/'
    csv_logger_cb = CSVLogger(csv_path + title + '.csv', append=False)
    save_results_cb = SaveResultsCallback() 

    # Obtain input 
    if not single_input:
        (X_train, y_train, X_test, y_test) = get_input_train_and_test()
    else:
        (X, y) = get_single_input()

    # Train the model
    if not single_input:
        history_cnn = cnn.fit(X_train, y_train, epochs=epochs, verbose=1, \
            callbacks=[csv_logger_cb, save_results_cb], validation_data=(X_test, y_test))
    else:
        history_cnn = cnn.fit(X, y, epochs=epochs, verbose=1, \
            callbacks=[csv_logger_cb, save_results_cb], validation_split=val_split)
    
    # Locally save the weights and entire model
    if save_model: 
        cnn.save_weights("model/weights/weights_" + title  +'.model')
        cnn.save("model/model/entire_model_" + title + '.hdf5')
    
    # Evalute
    score = cnn.evaluate(X_test, y_test)
    print(score)

    # Save results
    get_results(history_cnn.history)
 
if __name__ == '__main__':
    main()
