from random import randint
import numpy as np
from matplotlib import pyplot as plt

from net import HopfieldNetwork
from trainers import train
from letters import L
from functions import get_sample

L = get_sample(L, 4)

for l in range(len(L)):
    plt.subplot(4, 3, 1 + 3*l)
    to_show = []
    to_show.append(L[l].reshape(7, 5))
    plt.imshow(to_show[-1], cmap='gray')

network = HopfieldNetwork(35)
train(network, L)

Test = []
for l in range(len(L)):
    Test.append(L[l])
    for pixel in range(2):
        Test[l][randint(0, 34)] *= -1

Result = []
for l in range(len(L)):
    Result.append(network.run(Test[l]))
    Result[l].shape = (7, 5)
    Test[l].shape = (7, 5)

for l in range(len(L)):
    plt.subplot(4, 3, 3*l + 2)
    plt.imshow(Test[l], cmap='gray')
    plt.subplot(4, 3, 3*l + 3)
    plt.imshow(Result[l], cmap='gray')

plt.show()
