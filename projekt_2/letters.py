from random import randint

import numpy as np
from matplotlib import pyplot as plt

from net import HopfieldNetwork
from trainers import hebbian_training
from temp import L

input_patterns = np.array([letter.flatten() for letter in L])

# Create the neural network and train it using the training patterns
network = HopfieldNetwork(35)

hebbian_training(network, input_patterns)

# Create the test patterns by using the training patterns and adding some noise to them
# and use the neural network to denoise them
a_test = a_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    a_test[p] *= -1

a_result = network.run(a_test)

a_result.shape = (7, 5)
a_test.shape = (7, 5)

u_test = u_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    u_test[p] *= -1

u_result = network.run(u_test)

u_result.shape = (7, 5)
u_test.shape = (7, 5)

t_test = t_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    t_test[p] *= -1

t_result = network.run(t_test)

t_result.shape = (7, 5)
t_test.shape = (7, 5)

s_test = s_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    s_test[p] *= -1

s_result = network.run(s_test)

s_result.shape = (7, 5)
s_test.shape = (7, 5)

# Show the results
plt.subplot(4, 2, 1)
plt.imshow(a_test, cmap='gray')
plt.subplot(4, 2, 2)
plt.imshow(a_result, cmap='gray')

plt.subplot(4, 2, 3)
plt.imshow(u_test, cmap='gray')
plt.subplot(4, 2, 4)
plt.imshow(u_result, cmap='gray')

plt.subplot(4, 2, 5)
plt.imshow(t_test, cmap='gray')
plt.subplot(4, 2, 6)
plt.imshow(t_result, cmap='gray')

plt.subplot(4, 2, 7)
plt.imshow(s_test, cmap='gray')
plt.subplot(4, 2, 8)
plt.imshow(s_result, cmap='gray')

plt.show()
