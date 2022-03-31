import random
import time
import numpy as np
import pandas as pd 
import torch
import keras
from torch import optim
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2
from get_random_letter import get_random_letter

word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
}
 
def letter_frequency(y):
    counter = [0 for i in range(26)]
    for letter in y:
        counter[int(letter)] += 1
    plot = plt.bar([i for i in range(26)], counter)
    plt.show()
 
def print_sample_train_letter(shuffled):
    _, ax = plt.subplots(4, 4, figsize = (10, 10))
    axes = ax.flatten()
 
    for i in range(16):
        shu = cv2.threshold(shuffled[i], 30, 200, cv2.THRESH_BINARY)
        axes[i].imshow(np.reshape(shuffled[i], (28, 28)), cmap = 'Greys')
    plt.show()
    
def print_test_letters_and_pred(model, x_test, categorical_test):
    _, axs = plt.subplots(4, 4, figsize = (10, 10))
    axs = axs.flatten()

    for i in range(16):
        axs[i].imshow(np.reshape(x_test[i], (28, 28)),'Greys')

        prediction = word_dict[np.argmax(categorical_test[i])]
        axs[i].set_title("Prediction: " + prediction, fontsize = 18)
    plt.show()