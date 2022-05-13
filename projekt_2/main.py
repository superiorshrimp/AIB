from random import randint
from matplotlib.cbook import flatten
import numpy as np
from matplotlib import pyplot as plt
import copy

from net import HopfieldNetwork
from modern_net import ModernHopfieldNetwork
from trainers import train
from letters import L
from functions import get_sample

n = 8

"""
L = get_sample(L, n)

for l in range(len(L)):
    plt.subplot(n, 3, 1 + 3*l)
    to_show = []
    to_show.append(L[l].reshape(7, 5))
    plt.imshow(to_show[-1], cmap='gray')

network = HopfieldNetwork(35)
train(network, L)

Test = []
for l in range(len(L)):
    Test.append(L[l])
    for pixel in range(5):
        Test[l][randint(0, 34)] *= -1

Result = []
for l in range(len(L)):
    Result.append(network.run(Test[l]))
    Result[l].shape = (7, 5)
    Test[l].shape = (7, 5)

for l in range(len(L)):
    plt.subplot(n, 3, 3*l + 2)
    plt.imshow(Test[l], cmap='gray')
    plt.subplot(n, 3, 3*l + 3)
    plt.imshow(Result[l], cmap='gray')

plt.show()
"""
L = get_sample(L, n)

for l in range(n):
    plt.subplot(n, 3, 1 + 3*l)
    to_show = []
    to_show.append(L[l].reshape(7, 5))
    plt.imshow(to_show[-1], cmap='gray')

network = ModernHopfieldNetwork(35)
network.set_patterns(L)

Test = []
for l in range(n):
    test = copy.deepcopy(L[l])
    test.flatten()
    print(test)
    Test.append(test)
    for pixel in range(5):
        Test[l][randint(0, 34)] *= -1

Result = []
for l in range(n):
    Result.append(network.run(Test[l]))
    Result[l].shape = (7, 5)
    Test[l].shape = (7, 5)
    print(Test[l])
    print(Result[l])

for l in range(n):
    plt.subplot(n, 3, 3*l + 2)
    plt.imshow(Test[l], cmap='gray')
    plt.subplot(n, 3, 3*l + 3)
    plt.imshow(Result[l], cmap='gray')

plt.show()
# """