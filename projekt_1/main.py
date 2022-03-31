#link do zbioru danych: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format
 
#imports
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
from printData import print_sample_train_letter
from printData import print_test_letters_and_pred
from createData import getData
from modelBuild import build_model

 
data_path = "D:\Data\A_ZHandwrittenData.csv" # "C:\\Users\\monik\\OneDrive\\Pulpit\\AIB2\\AIB\\data.csv"

x_train, x_test, categorical_train, categorical_test, shuff = getData("D:\Data\A_ZHandwrittenData.csv")

print_sample_train_letter(shuff)

my_model = build_model(x_train, x_test, categorical_train, categorical_test)

print_test_letters_and_pred(my_model, x_test, categorical_test)
