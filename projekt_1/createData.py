
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
    df = pd.read_csv(data_path, dtype = np.float32)
    df = df.sample(frac = 0.1)
    
    
    x = df.drop('0', axis = 1)
    y = df['0']

    
    letter_frequency(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
    x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))
    
    
    shuff = shuffle(x_train[:100])

    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    
    
    categorical_train = to_categorical(y_train, num_classes = 26, dtype = 'int')
    
    categorical_test = to_categorical(y_test, num_classes = 26, dtype = 'int')

    return x_train, x_test, categorical_train, categorical_test, shuff
    