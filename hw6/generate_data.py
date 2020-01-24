import numpy as np
from numpy.linalg import svd
import sys

def generate_data(N,p,condition_number):
    data = np.random.rand(N,p)
    u, s, vt = svd(data)
    p = len(s)
    s = np.linspace(1,condition_number,p)
    data = u[:,:p] @ np.diag(s) @ vt[:p,]
    b = np.random.rand(N)
    np.save("A_{}.npy".format(condition_number),data)
    np.save("b_{}.npy".format(condition_number),b)


if len(sys.argv) < 4:
    print("N p Condition_Number")
else:
    generate_data(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
