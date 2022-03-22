#link do zbioru danych: https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

import numpy as np
import pandas as pd 
import torch
from torch import optim
import torch.nn as nn
from PIL import Image
import seaborn as sns
import os 

data_path = "D:\data\project_1\data.csv"

'''
df = pd.read_csv(data_path, dtype = np.float32)
df = df.sample(frac=1)

X = df.iloc[:,1:].values / 255 
y = df.iloc[:,0].values

fig,ax = plt.subplots(2,5)
for i in range(10):
    nparray = X[i].reshape(28,28)
    image = Image.fromarray(nparray * 255)
    ax[i%2][i//2].imshow(image)
fig.show()
'''