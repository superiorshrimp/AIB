from xml.etree.ElementTree import C14NWriterTarget
import math
import numpy as np
import random
from F_goal import F_goal
import matplotlib.pyplot as plt

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

def draw_map(W):
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.scatter(x = [el[1] for el in W], y = [el[2] for el in W], s = [el[0] for el in W])
    plt.show()
    
def draw_results(W, v):
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.scatter(x = [el[1] for el in W], y = [el[2] for el in W], s = [el[0] for el in W])
    plt.scatter(x = [el for el in v[::2]], y = [el for el in v[1::2]])
    plt.show()

def draw_state(W, v):
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.scatter(x = [el[1] for el in W], y = [el[2] for el in W], s = [el[0] for el in W])
    plt.scatter(x = [el for el in v[::2]], y = [el for el in v[1::2]])
    plt.show()

def set_parameters(no_area = 10, first_location = 7, neighbour = 3, no_iteration = 1000, no_tries = 100):
    #no_area = ilość przeszukiwanych obszarów (populacja pszczół)
    #first_location = okres losowania rozmieszczenia pojemników (zalecane 1-10)
    #neighbour = wielkość sąsiedztwa przeszukiwania
    #no_iteration = ilość iteracji
    #no_tries = ilość prób zmian jednego wektora
    return no_area, first_location, neighbour, no_iteration, no_tries

def create_matrix_c(no_area, first_location):
    C = np.zeros( (no_area, 6) )
    for i in range(no_area):
        for j in range(6):
            C[i][j] = first_location*int(random.randrange(1,10))     
    return C

def find_max_F_idx(F):
    K = [(F[i],i) for i in range(len(F))]
    K.sort()
    K.reverse()
    return K[0][1]

def find_next_idx(F,which):
    K = [(F[i],i) for i in range(len(F))]
    K.sort()
    K.reverse()
    
    i = 0
    while K[i][1] not in which:
        i += 1
     
    return K[i][1]

def change_vector_c_with_neighbour(neighbour, vec):
    for i in range(len(vec)):
        d = int(random.randrange(-neighbour,neighbour+1))
        vec[i] = vec[i] + d
        if vec[i]<1:
            vec[i] = 1
        elif vec[i]>100:
            vec[i] = 100
    return vec

def find_solution(neighbour, no_area, no_iteration, vector_c,F_old,idx):
    which_time = 0
    which_solution = [idx]

    for i in range(no_iteration):
        change_vector_c = change_vector_c_with_neighbour(neighbour, vector_c)
        F_new = F_goal(W,change_vector_c,no_area,1)
        
        if F_old < F_new:
            F_old = F_new
            vector_c = change_vector_c
        else:
            which_time += 1
        
        if which_time > no_iteration//2:
            which_time = 0
            idx = find_next_idx(F,which_solution)
            which_solution.append(idx)
            vector_c = C[idx]
            
        #if i%100 == 0:
        #    draw_state(W, vector_c)
    
    return F_old, vector_c

if __name__ == "__main__":
    no_area, first_location, neighbour, no_iteration, no_tries = set_parameters()
    draw_map(W)
    
    C = create_matrix_c(no_area, first_location)
    F = F_goal(W,C,no_area,0)
    idx = find_max_F_idx(F)
    F_old = F[idx]
    vector_c = C[idx]
    
    F_old, vector_c = find_solution(neighbour, no_area, no_iteration, vector_c,F_old,idx)
    
    print(F_old)
    draw_results(W, vector_c)