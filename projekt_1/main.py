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
#from keras.optimizers import SGD, Adam
#from tensorflow.keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2
from get_random_letter import get_random_letter
 
def print_sample_letters(X):
    _, plot = plt.subplots(2, 3)
    for i in range(6):
        plot[i % 2][i // 2].imshow(Image.fromarray(255 * X[i].reshape(28, 28)))
    plt.show()
 
def letter_frequency(y):
    counter = [0 for i in range(26)]
    for letter in y:
        counter[int(letter)] += 1
    plot = plt.bar([i for i in range(26)], counter)
    plt.show()
    
 
# X = df.iloc[:,1:].values / 255
# y = df.iloc[:,0].values
#data_path = "D:\data\project_1\data.csv"
data_path = "C:\\Users\\Szymon\\AIB\\projekt_1\\data_mini.csv"
df = pd.read_csv(data_path, dtype = np.float32)
df = df.sample(frac = 0.1)
 
 
x = df.drop('0', axis = 1)
y = df['0']
 
#print_sample_letters(x)
 
#letter_frequency(y)
 
word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
}
 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))
 
 
plt.style.use('fivethirtyeight')
plt.xkcd()
 
shuff = shuffle(x_train[:100])
fig, ax = plt.subplots(3, 3, figsize = (15, 15))
axes = ax.flatten()
 
'''
for i in range(9):
    shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap = 'Greys')
plt.show()
'''
 
 
 
print('Train Data Shape:', x_train.shape)
print('Test Data Shape:', x_test.shape)
 
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
print("New shape of train data:", x_train.shape)
 
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print("New shape of test data:", x_test.shape)
 
 
categorical_train = to_categorical(y_train, num_classes = 26, dtype = 'int')
print("New shape of train labels:", categorical_train.shape)
 
categorical_test = to_categorical(y_test, num_classes = 26, dtype = 'int')
print("New shape of test labels:", categorical_test.shape)
 
 
my_model = Sequential()
 
 
my_model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
my_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
 
my_model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
my_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
 
my_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'valid'))
my_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))
 
my_model.add(Flatten())
 
my_model.add(Dense(64, activation = "relu"))
my_model.add(Dense(128, activation = "relu"))
 
my_model.add(Dense(26, activation = "softmax"))
 
my_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = my_model.fit(x_train, categorical_train, epochs = 1, validation_data = (x_test, categorical_test))
 
my_model.save("C:\\Users\\Szymon\\AIB\\projekt_1") #'C:\\Users\\monik\\OneDrive\\Pulpit\\AIB\\AIB\\projekt_1')

letter = get_random_letter()
y, x = letter[0], letter[1:]
img = np.reshape(x, (28,28))
predictions = my_model.predict(x)
print(y)
print(predictions)

# for i, ax in enumerate(axes):
#     img = np.reshape(im2, (28, 28))
#     ax.imshow(img, cmap = 'Greys')
    
#     pred = word_dict[np.argmax(im2)]
#     ax.set_title("Prediction: " + pred, fontsize = 20, fontweight = 'bold', color = 'red')
#     ax.grid()
 
# plt.show()
 
# model = keras.models.load_model('C:\\Users\\monik\\OneDrive\\Pulpit\\AIB\\AIB\\projekt_1\\saved_model.pb')