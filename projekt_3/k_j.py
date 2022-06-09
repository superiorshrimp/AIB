from math import sqrt

def k_j(Cx,Cy,W,i):
    s = (W[i][1] - Cx)**2 + (W[i][2] - Cy)**2
    return (W[i][0] * 141.42) / (20 * sqrt(s) + 0.0001)