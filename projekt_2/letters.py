from random import randint

import numpy as np
from matplotlib import pyplot as plt

from net import HopfieldNetwork
from trainers import hebbian_training

# Create the training patterns
a_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])

b_pattern = np.array([[1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 1, 0, 0],
                      [1, 0, 0, 1, 0],
                      [1, 0, 0, 1, 0],
                      [1, 1, 1, 0, 0]])

c_pattern = np.array([[0, 0, 1, 1, 1],
                      [0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1],])

d_pattern = np.array([[1, 1, 1, 0, 0],
                      [1, 0, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 1, 0],
                      [1, 1, 1, 0, 0],])

e_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],])

f_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],])

g_pattern = np.array([[0, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [0, 1, 1, 1, 0],])

h_pattern = np.array([[1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],])

i_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],])

j_pattern = np.array([[0, 1, 1, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0],
                      [0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 0],])


u_pattern = np.array([[1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]])

t_pattern = np.array([[1, 1, 1, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])

s_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1]])

a_pattern *= 2
a_pattern -= 1

b_pattern *= 2
b_pattern -= 1

c_pattern *= 2
c_pattern -= 1

d_pattern *= 2
d_pattern -= 1

e_pattern *= 2
e_pattern -= 1

f_pattern *= 2
f_pattern -= 1

g_pattern *= 2
g_pattern -= 1

h_pattern *= 2
h_pattern -= 1

i_pattern *= 2
i_pattern -= 1

j_pattern *= 2
j_pattern -= 1

u_pattern *= 2
u_pattern -= 1

t_pattern *= 2
t_pattern -= 1

s_pattern *= 2
s_pattern -= 1

input_patterns = np.array([a_pattern.flatten(),
                           b_pattern.flatten(),
                           c_pattern.flatten(),
                           d_pattern.flatten(),
                           e_pattern.flatten(),
                           f_pattern.flatten(),
                           g_pattern.flatten(),
                           h_pattern.flatten(),
                           i_pattern.flatten(),
                           u_pattern.flatten(), 
                           t_pattern.flatten(),
                           s_pattern.flatten()])

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
