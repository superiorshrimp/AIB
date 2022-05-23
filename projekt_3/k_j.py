import math

def k_j(Cx,Cy,W,i):
    sum1 = (W[i][1]-Cx)**2+(W[i][2]-Cy)**2
    return (W[i][0]*141.42)/(20*math.sqrt(sum1)+0.0001)