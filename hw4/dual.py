from prox import prox_dp
import numpy as np
import sys
from math import inf
from scipy.optimize import linprog from matplotlib import pyplot as plt

def f(x,u,t,lamb):
    if np.any(x <= 0) or np.any((1-x) <= 0) or np.any((lamb-u) <= 0) or np.any((lamb+u) <= 0):
        return inf
    else:
        return np.dot(x,np.log(x)) + np.dot(1-x, np.log(1-x)) - 1/t * np.sum(np.log(x) + np.log(1-x)) - 1/t * np.sum(np.log(lamb-u)+np.log(lamb+u))

def loss(u,D,y,lamb,t):
    yDTu =  y * np.dot(np.transpose(D),u)
    return f(yDTu,u,t,lamb)

def gradient(D,u,y,t,lamb):
    yDTu = y * np.dot(np.transpose(D),u)
    fprime = np.log(yDTu)  - np.log( 1 - yDTu) - 1/t * 1/yDTu + 1/t * 1/(1-yDTu)
    gprime = y.reshape(-1,len(y)) * D
    grad_part = np.dot(gprime,fprime)
    return grad_part - 1/t * (-1/(lamb-u) + 1/(lamb+u))

def returnFeasible(y,D,lamb,delta):
    n = len(y)
    Aub = np.concatenate((y.reshape(n,1) * np.transpose(D), - y.reshape(n,1) * np.transpose(D)))
    bub = np.zeros(Aub.shape[0])
    bub[:n] = 1 - delta
    bub[n:] = -delta
    bnds = [(delta-lamb ,lamb-delta) for i in range(len(y)-1)]
    c = np.zeros_like(y[1:])
    res = linprog(c,A_ub=Aub,b_ub=bub,bounds=bnds)
    print(res)
    return res.x 

def backtracking(D,u,y,tau,lamb,beta):
    t=1
    grad = gradient(D,u,y,tau,lamb)
    norm2grad = np.dot(grad,grad)
    currLoss = loss(u,D,y,lamb,tau)
    iterations = 0
    while True:
        iterations = iterations + 1
        lhs = loss(u-t*grad,D,y,lamb,tau)
        rhs = currLoss - t/2 * norm2grad
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t, iterations

def training(u,y,D,beta=0.8,lamb=20,tau=1e3,tol=1e-6,maxiter=50000):
    u_new = np.zeros_like(u)
    n = len(y)
    losses = [loss(u,D,y,lamb,tau)]
    iterations = 0
    while True:
        iterations = iterations + 1
        grad = gradient(D,u,y,tau,lamb)
        t,i = backtracking(D,u,y,tau,lamb,beta)
        u_new = u - t * grad
        iterations = iterations + i
        losses.append(loss(u_new,D,y,lamb,tau))
        if np.abs(losses[-1]-losses[-2]) < tol or iterations >= maxiter: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            u = u_new
    return u, losses

def loss_primal(theta,z,lamb):
    return -np.dot(z,theta) + np.sum(np.log(1+np.exp(theta))) + lamb * np.sum(np.abs(theta[1:] - theta[:-1]))

def loss_primal_g(theta,z):
    return -np.dot(z,theta) + np.sum(np.log(1+np.exp(theta)))

def gradient_primal(theta,z):
    return -z + np.exp(theta)/(1+np.exp(theta))

def prox_opt(theta,lamb):
    tmp = np.zeros_like(theta)
    prox_dp(len(theta),theta,lamb,tmp)
    return tmp 

def backtracking_primal(curr,grad,z,lamb,beta):
    t=1
    def G(t_x):
        return (curr - prox_opt(curr-t_x*grad,lamb))/t_x
    while True:
        G_curr = G(t)
        rhs = loss_primal_g(curr,z) - t * np.dot(grad,G_curr) + t/2 * np.dot(G_curr,G_curr)
        lhs = loss_primal_g(curr - t*G_curr,z)
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t

def training_primal(theta,z,beta=0.8,lamb=20,t=1,tol=1e-6):
    theta_tmp = np.copy(theta)
    theta_new = np.zeros_like(theta)
    n = len(z)
    losses = [loss_primal(theta_tmp,z,lamb)]
    while True:
        grad = gradient_primal(theta_tmp,z)
        t = backtracking_primal(theta_tmp,grad,z,lamb,beta)
        theta_new = prox_opt(theta_tmp - t * grad,lamb)
        losses.append(loss_primal(theta_new,z,lamb))
        if np.abs(losses[-1]-losses[-2]) < tol: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            theta_tmp = np.copy(theta_new)
    return theta_new, losses

def main():
    data = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    theta = np.random.rand(len(data))
    theta_new, losses_primal = training_primal(theta,data)
    n = len(data)
    y = np.copy(data)
    y[y==0] = -1
    t = 1e3
    D = np.zeros((n-1,n))
    D[range(n-1),range(n-1)] = 1                                                                                                                    
    D[range(n-1),range(1,n)] = -1
    lamb = 20
    delta = 0.01
    u = returnFeasible(y,D,lamb,delta)
    u_new, losses_dual = training(u,y,D,maxiter=50000)
    DTu = np.dot(np.transpose(D), u_new)
    theta_u = - y * np.log(DTu/(y - DTu))
    print(losses_primal[-1], losses_dual[-1])
    plt.rcParams.update({'font.size'   : 22})
    plt.plot(range(len(data)), data,"o",label="Data")
    plt.plot(range(len(theta_new)),np.exp(theta_new)/(1+np.exp(theta_new)),"o",label="primal solution")
    plt.plot(range(len(theta_u)),np.exp(theta_u)/(1+np.exp(theta_u)),"o",label="dual solution")
    plt.legend()#fontsize="xx-large")
    plt.xlabel("Timepoint i")
    plt.show()
    plt.plot(range(len(losses_primal)),losses_primal)
    plt.plot(range(len(losses_dual)),-np.array(losses_dual))
    plt.show()


main()
