import numpy as np
from random import randint, shuffle


class HopfieldNetwork(object):
    def __init__(self, neurons_num):
        self._neutrons_num = neurons_num
        self._weights = np.random.uniform(-1.0, 1.0, (neurons_num, neurons_num))

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    def calculate_neuron_output(self, neuron, pattern):
        neurons_num = len(pattern)

        s = 0.0

        for j in range(neurons_num):
            s += self._weights[neuron][j] * pattern[j]

        res = 1.0 if s > 0.0 else -1.0

        return res

    def run(self, pattern, max_iterations=10):
        counter = 0

        res = pattern.copy()

        while True:
            update = range(self._neutrons_num)
            shuffle(list(update))

            changed, res = self.run_once(update, res)

            counter += 1

            if not changed or counter == max_iterations:
                return res

        
    def run_once(self, update, pattern):
        res = pattern.copy()

        flag = False
        for n in update:
            n_output = self.calculate_neuron_output(n, res)
            if n_output != res[n]:
                flag = True
                res[n] = n_output

        return flag, res

