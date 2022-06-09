from k_j import k_j
import numpy as np

def K_m(W,C,i,no_area,flag):
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    suma = np.zeros(no_area)
    if flag == 0:
        for j in range(no_area):
            sum_1 = k_j(C[j][0],C[j][1],W,i)
            sum_2 = k_j(C[j][2],C[j][3],W,i)
            sum_3 = k_j(C[j][4],C[j][5],W,i)

            suma[j] = sum_1 + sum_2 + sum_3
            if suma[j] > W[i][0]:
                suma[j] = W[i][0] 
    else:
        sum_1 = k_j(C[0],C[1],W,i)
        sum_2 = k_j(C[2],C[3],W,i)
        sum_3 = k_j(C[4],C[5],W,i)

        suma = sum_1 + sum_2 + sum_3
        if suma > W[i][0]:
            suma = W[i][0]

    return suma