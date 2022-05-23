import math

def k_j(Cx,Cy,W,i):
    return ((W[i][0]*141.42)/(20*math.sqrt((W[i][1])-Cx)^2+(W[i][2]-Cy)^2)+0.0001)