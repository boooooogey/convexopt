from scipy.optimize import linprog
import numpy as np
import sys

def main():
    y = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    n = len(y)
    D = np.zeros((n-1,n))
    D[range(n-1),range(n-1)] = 1
    D[range(n-1),range(1,n)] = -1
    delta = 0.01
    lamb = 20
    Aub = np.concatenate((y.reshape(n,1) * np.transpose(D), - y.reshape(n,1) * np.transpose(D)))
    bub = np.zeros(Aub.shape[0])
    bub[:n-1] = 1 - delta
    bub[n-1:] = delta
    bnds = [(None,lamb - delta) for i in range(len(y)-1)]
    c = np.zeros_like(y[1:])
    print(Aub.shape)
    print(len(c))
    print(len(bub))
    print(len(bnds))
    res = linprog(c,A_ub=Aub,b_ub=bub,bounds=bnds)
    print(res.x)
    print(np.all(np.dot(Aub,res.x)<= bub))
    print(np.all(res.x<= lamb-delta))
    #print(Aub)
    #print(bub)

main()
