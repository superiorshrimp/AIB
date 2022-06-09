from K_m import K_m
import numpy as np

def F_goal(W,C,no_area,flag):
    if flag == 0:
        sum_denominator = 0
        matrix_for_all_solution = np.zeros(no_area)
        for i in range(12):
            sum_denominator += W[i][0]

        for i in range(12):
            new_vector = K_m(W,C,i,no_area,0)
            for j in range(no_area):
                matrix_for_all_solution[j] += new_vector[j]
            
        F = np.zeros(no_area)
        for i in range(no_area):
            F[i] = matrix_for_all_solution[i] * 100 / sum_denominator
            
    else:
        sum_denominator = 0
        matrix_for_all_solution = 0
        for i in range(12):
            sum_denominator += W[i][0]
        
        for i in range(12):
            matrix_for_all_solution += K_m(W,C,i,no_area,1)

        F = matrix_for_all_solution * 100 / sum_denominator
        
    return F