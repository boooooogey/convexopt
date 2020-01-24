import numpy as np
from time import time
from matplotlib import pyplot as plt
from IPython import embed
from frank_wolfe import *

def main():
    shape = (20,20)
    ratio_missing = 0.3
    m = int(shape[0] * shape[1] * ratio_missing)
    print("Shape of the matrix: {}.".format(shape))
    print("Number of missing entries: {}.".format(m))
    mat = np.random.rand(shape[0],shape[1])
    mask = np.ones(shape[0]*shape[1],dtype=bool)
    mask[:m] = False
    np.random.shuffle(mask)
    mask = mask.reshape(shape)
    Y = np.random.rand(shape[0],shape[1])
    Y[np.logical_not(mask)] = 0
    t = trace_norm(Y) * np.linspace(0.1,1,3)
    t = [float(int(i)) for i in t]
    print("t {}".format(t))
    p = min(shape[0],shape[1])
    u, ss, vt = np.linalg.svd(Y)
    iterations = 10000
    time_alg = {"fw":[],"fw_b":[],"pgd":[],"pgd_a":[]}
    for i in t:
        r = np.linalg.matrix_rank(Y)
        B_init = np.zeros(shape)
        s = time()
        out_fw = frank_wolfe(B_init,Y,mask,i,tol=0, max_iter = iterations)
        e = time()
        print("Frank Wolfe finished.({})".format(e-s))
        time_alg["fw"].append(e-s)
        s = time()
        out_fw_b = frank_wolfe(B_init,Y,mask,i,tol=0, use_backtracking=True, max_iter = iterations)
        e = time()
        print("Frank Wolfe finished.({})".format(e-s))
        time_alg["fw_b"].append(e-s)
        s = time()
        out_pgd = projected_gradient_descent(B_init, Y, mask, i, tol=0, max_iter = iterations)
        e = time()
        time_alg["pgd"].append(e-s)
        print("Projected gradient descent finished.({})".format(e-s))
        #s = time()
        #out_pgd_b = projected_gradient_descent(B_init, Y, mask, i, r, tol=0, max_iter = iterations,use_backtracking=True)
        #e = time()
        #print("Projected gradient descent with backtracking finished.({})".format(e-s))
        s = time()
        out_pgd_a = projected_gradient_descent(B_init, Y, mask, i, tol=0, max_iter = iterations,use_acceleration=True)
        e = time()
        print("Projected gradient descent with acceleration finished.({})".format(e-s))
        time_alg["pgd_a"].append(e-s)
        #print(loss(out_fw[0],out_pgd[0],mask))
        plt.semilogy(np.arange(len(out_fw[1])),out_fw[1],label='frank wolfe')
        print(trace_norm(out_fw[0]))
        plt.semilogy(np.arange(len(out_fw_b[1])),out_fw_b[1],label='frank wolfe with backtracking')
        print(trace_norm(out_fw_b[0]))
        plt.semilogy(np.arange(len(out_pgd[1])),out_pgd[1],label='projected gradient descent')
        print(trace_norm(out_pgd[0]))
        #plt.semilogy(np.arange(len(out_pgd_b[1])),out_pgd_b[1],label='projected gradient descent with backtracking')
        #print(trace_norm(out_pgd_b[0]))
        plt.semilogy(np.arange(len(out_pgd_a[1])),out_pgd_a[1],label='projected gradient descent with acceleration')
        print(trace_norm(out_pgd_a[0]))
        plt.ylabel("Objective")
        plt.xlabel("Iterations")
        plt.title("Trace Norm {}".format(i))
        plt.legend()
        plt.show()
    embed()

main()
