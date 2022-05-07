import random
import numpy as np

def get_sample(T, k):
    indexes = [i for i in range(len(T))]
    indexes = random.sample(indexes, k)
    ret = []
    for id in indexes:
        ret.append(T[id].flatten())
    return np.array(ret)