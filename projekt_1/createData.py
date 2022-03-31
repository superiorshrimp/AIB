
import random
import numpy as np
import pandas as pd 
import keras
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2
from printData import letter_frequency


def getData(data_path):
    data_handler = pd.read_csv(data_path, dtype = np.float32)
    data_handler = data_handler.sample(frac = 0.1)
    
    
    x = data_handler.drop('0', axis = 1)
    y = data_handler['0']

    
    letter_frequency(y)

    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size = 0.2)
    x_train_data = np.reshape(x_train_data.values, (x_train_data.shape[0], 28, 28))
    x_test_data = np.reshape(x_test_data.values, (x_test_data.shape[0], 28, 28))
    
    
    shuff = shuffle(x_train_data[:100])

    
    x_train_data = x_train_data.reshape(x_train_data.shape[0], x_train_data.shape[1], x_train_data.shape[2], 1)

    x_test_data = x_test_data.reshape(x_test_data.shape[0], x_test_data.shape[1], x_test_data.shape[2], 1)
    
    
    categorical_train = to_categorical(y_train_data, num_classes = 26, dtype = 'int')
    
    categorical_test = to_categorical(y_test_data, num_classes = 26, dtype = 'int')

    return x_train_data, x_test_data, categorical_train, categorical_test, shuff
    