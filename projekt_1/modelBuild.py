import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2



def build_model(x_train, x_test, categorical_train, categorical_test):
    model = Sequential()
    
    
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

    model.add(Flatten())

    model.add(Dense(64, activation = "relu"))
    model.add(Dense(128, activation = "relu"))
    
    model.add(Dense(26, activation = "softmax"))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(x_train, categorical_train, epochs = 10, validation_data = (x_test, categorical_test))

    model.save(("C:\\Users\\Admin\\Desktop\\Studia\\IVsemestr\\Biol\\AIB\\projekt_1\\model.h5"))  #('C:\\Users\\monik\\OneDrive\\Pulpit\\AIB2\\AIB\\projekt_1')

    return model