import numpy as np
from random import randint, shuffle
import math            
from numpy import vectorize, array, dot, sum
import random
import copy


class ModernHopfieldNetwork(object):
    def __init__(self, neurons_num):
        self._neutrons_num = neurons_num
        self._patterns = None
    
    def set_patterns(self, patterns):
        print(patterns)
        self._patterns = patterns

    def run_once(self, pattern):
        sgn = vectorize(lambda x: -1 if x<0 else +1)
        xPatterns = copy.deepcopy(pattern)
        yPatterns = copy.deepcopy(pattern)
        randomNumber = random.sample(range(0,len(yPatterns)), len(yPatterns))
        print(randomNumber)

        for x in range(0, len(randomNumber)):
            xPatterns[randomNumber[x]] = 1
            e1 = self.calculate_hopfield_energy(xPatterns)
            xPatterns[randomNumber[x]] = -1
            e2 = self.calculate_hopfield_energy(xPatterns)
            print(randomNumber[x], e1, e2)
            xPatterns[randomNumber[x]] = pattern[randomNumber[x]]
            yPatterns[randomNumber[x]] = sgn( -1 * e1 + e2)
            print(self.calculate_hopfield_energy(yPatterns))

        return yPatterns

    
    def run(self, pattern, max_iterations=1):
        counter = 0

        res = pattern.copy()

        while True:
            res = self.run_once(res)

            counter += 1

            if counter == max_iterations:
                return res


    def calculate_hopfield_energy(self, pattern):
        e = array([math.exp(dot(p,pattern)) for p in self._patterns])
        return sum(e) * -1