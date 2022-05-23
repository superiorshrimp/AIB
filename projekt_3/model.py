
from xml.etree.ElementTree import C14NWriterTarget
import math
import numpy as np
import random
from F_goal import F_goal

no_area = int(input('Podaj ilosc przeszukiwanych obszarow (populacja pszczol): '))
first_location = int (input('Podaj okres losowania rozmieszczenia pojemnikow (1-10 zalecane): '))
neighbour = int(input('Podaj wielkosc sasiedztwa przeszukiwania: '))
no_iteration = int(input('Podaj ilosc iteracji: '))

print(no_area)

W = [[100, 25, 65],
    [200, 23, 8],
    [327, 7, 13],
    [440, 95, 53],
    [450, 95, 53],
    [639,54, 56],
    [650, 67, 78],
    [678, 32, 4],
    [750, 24, 76],
    [801, 66, 89],
    [945, 84, 4],
    [967, 34, 23]]


def create_matric_c():
    C = np.zeros( (no_area, 6) )
    for i in range(no_area):
            for j in range(6):
                C[i][j] = first_location*int(random.randrange(1,10))
                
    return C

def find_max_F_idx(F):
    K = [0]*len(F)
    
    for i in range(len(F)):
       K[i] = ((F[i],i))
    
    K.sort()
    K.reverse()
    return K[0][1]

def change_vector_c_with_neighbour(vec):
    for i in range(len(vec)):
        d = int(random.randrange(-neighbour,neighbour))
        vec[i] = vec[i] + d
        if (vec[i]< 1): vec[i] = 1
        if (vec[i]>100): vec[i] = 100
    return vec

C = create_matric_c()
# print(C)
F = F_goal(W,C,no_area,0)
# print(F)
idx = find_max_F_idx(F)
# print(idx)
F_old = F[idx]
vector_c = C[idx]
# print(vector_c)

for i in range(no_iteration):
    change_vector_c = change_vector_c_with_neighbour(vector_c)
    F_new = F_goal(W,change_vector_c,no_area,1)
    # print(F_old)
    # print(F_new)

    if F_old < F_new:
        F_old = F_new
        vector_c = change_vector_c
        

print(F_old)