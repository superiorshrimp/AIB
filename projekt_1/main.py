#link do zbioru danych: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

#imports
import time
import numpy as np
import pandas as pd 
import torch
from torch import optim
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import os
from torch.utils.data import Dataset

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

#data_path = "D:\data\project_1\data.csv"
data_path = "./projekt_1/data_mini.csv"
df = pd.read_csv(data_path, dtype = np.float32)
df = df.sample(frac = 0.1)

X = df.iloc[:,1:].values / 255
y = df.iloc[:,0].values

#print_sample_letters(X)
letter_frequency(y)