import cvxpy as cp
import numpy as np
import sys
import pickle as pkl
from matplotlib import pyplot as plt
from IPython import embed

def loss_cvx(A,b,x,lamb):                                                                                                                                
    return cp.norm2(cp.matmul(A,x)-b)**2 + lamb * cp.norm1(x)                                                                                      
                                                                                                                                                         
def solve_cvx(A,b,lamb):                                                                                                                                 
    p = A.shape[1]
    x = cp.Variable(p)                                                                                                                                 
    lambd = cp.Parameter(nonneg=True)                                                                                                                    
    lambd.value = lamb                                                                                                                                   
    problem = cp.Problem(cp.Minimize(loss_cvx(A, b, x, lambd)))                                                                                          
    problem.solve()
    return x, problem

def main():
    A = np.load("A.npy")
    b = np.load("b.npy")
    lamb = np.linspace(0,1,5)
    outs = dict()
    for i in lamb:
        outs[i] = solve_cvx(A,b,i)
    with open("admm_result.pkl",'rb') as file:
        distributed_out = pkl.load(file)
    #spasity
    for i in lamb:
        tmp = np.array(distributed_out[(5,4,i,5.0)][1]) #- outs[i][1].value
        plt.semilogy(range(len(tmp)),tmp,label="lamb = {}".format(i))
    plt.ylabel("Objective")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Varying Sparsity")
    plt.show()

    #lagrange
    rho = [3.0,4.0,5.0,6.0,7.0,8.0]
    for i in rho:
        tmp = np.array(distributed_out[(5,4,0.5,i)][1]) - outs[0.5][1].value
        plt.semilogy(range(len(tmp)),tmp,label="rho = {}".format(i))
    plt.ylabel("Optimality Gap")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Varying Lagrangian Parameter")
    plt.show()
    #subsets
    number_of_processes = [2,4,6,8]
    for i in number_of_processes:
        tmp = np.array(distributed_out[(5,i,0.5,5.0)][1]) - outs[0.5][1].value
        plt.semilogy(range(len(tmp)),tmp,label="number of subsets = {}".format(i))
    plt.ylabel("Optimality Gap")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Varying Subset Number")
    plt.show()
    #condition number 
    condition_numbers = [1, 5, 10]
    for i in condition_numbers:
        tmp = np.array(distributed_out[(i,4,0.5,5.0)][1])# - outs[0.5][1].value
        plt.semilogy(range(len(tmp)),tmp,label="condition number = {}".format(i))
    plt.ylabel("Objective")
    plt.xlabel("Iterations")
    plt.legend()
    plt.title("Varying Condition Number")
    plt.show()
    #print( np.sqrt((x.value-distributed_out[0]) @ (x.value-distributed_out[0])))
    #plt.semilogy(range(len(distributed_out[1])), np.array(distributed_out[1])-problem.value)
    #plt.ylabel("Optimality Gap")
    #plt.xlabel("Iterations")
    #plt.show()
    embed()

main()
