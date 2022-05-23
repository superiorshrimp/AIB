from k_j import k_j

def K_m(W,C,i,no_area,flag):
    sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
    suma = 0;
    if flag == 0:
        for j in range(no_area):
            sum_1 =  k_j(C[0,0],C[0,1],W,i)
            sum_2 =  k_j(C[0,2],C[0,3],W,i)
            sum_3 = k_j(C[0,4],C[0,5],W,i)

            suma = sum_1 + sum_2 + sum_3
            if suma >  W[i][0]:
                suma = W[i][0]
            
    else:
        sum_1 =  k_j(C[0,0],C[0,1],W,i)
        sum_2 =  k_j(C[0,2],C[0,3],W,i)
        sum_3 = k_j(C[0,4],C[0,5],W,i)


        suma = sum_1 + sum_2 + sum_3
        if suma >  W[i][0]:
            suma = W[i][0]


    return suma