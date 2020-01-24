from prox import prox_dp
import pickle
import os
import numpy as np
import sys
from math import inf
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import pylab
from IPython import embed
import time

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

def hessian(D,u,y,t,lamb):
    yDTu =  y * np.dot(np.transpose(D),u)
    fprime = 1/yDTu + 1/(1-yDTu) + 1/t * 1/np.power(yDTu,2) + 1/t * 1/np.power(1-yDTu,2)
    hessian = np.dot(fprime.reshape(1,-1) * D, np.transpose(D))
    diagonal = 1/t * ( 1/np.power(lamb+u,2) + 1/np.power(lamb-u,2) )
    hessian[range(hessian.shape[0]),range(hessian.shape[0])] = hessian[range(hessian.shape[0]),range(hessian.shape[0])] + diagonal
    return hessian


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

def backtracking(D,u,y,grad,hessinv,tau,lamb,beta):
    t=1
    v = -np.dot(hessinv,grad)
    norm2grad = np.dot(grad,v)
    currLoss = loss(u,D,y,lamb,tau)
    while True:
        lhs = loss(u+t*v,D,y,lamb,tau)
        rhs = currLoss + t/2 * norm2grad
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t

def barrier_method(u,y,D,tau0=5,mu=10,eps=1e-8,lamb=20,maxiter=100):
    tau_step = tau0
    u_step = u
    m = 4 * len(y) - 2
    iterations = 0
    steps = [u_step]
    while True:
        curr_center = centering(u_step,y,D,tau=tau_step,lamb=lamb)
        iterations = iterations + curr_center[1][-1]
        u_step = curr_center[0]
        steps.append(u_step)
        if m/tau_step < eps or iterations >= maxiter:
            break
        else:
            tau_step = tau_step * mu
    return u_step, iterations, steps
        
def centering(u,y,D,beta=0.8,lamb=20,tau=1e3,tol=1e-6,maxiter=50000):
    u_new = np.zeros_like(u)
    n = len(y)
    losses = [loss(u,D,y,lamb,tau)]
    iterations_step = [0] 
    iterations = 0
    steps = [u]
    while True:
        grad = gradient(D,u,y,tau,lamb)
        hessinv = np.linalg.inv(hessian(D, u, y, tau, lamb))
        t = backtracking(D,u,y,grad,hessinv,tau,lamb,beta)
        u_new = u - t * np.dot(hessinv,grad)
        iterations = iterations + 1
        steps.append(u_new)
        iterations_step.append(iterations)
        losses.append(loss(u_new,D,y,lamb,tau))
        if np.abs(losses[-1]-losses[-2]) < tol or iterations >= maxiter: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            u = u_new
    return u, iterations_step, steps

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
    iterations = 0
    def G(t_x):
        return (curr - prox_opt(curr-t_x*grad,lamb))/t_x
    while True:
        iterations = iterations + 1
        G_curr = G(t)
        rhs = loss_primal_g(curr,z) - t * np.dot(grad,G_curr) + t/2 * np.dot(G_curr,G_curr)
        lhs = loss_primal_g(curr - t*G_curr,z)
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t, iterations

def training_primal(theta,z,beta=0.8,lamb=20,t=1,tol=1e-6,maxiter=50000):
    theta_tmp = np.copy(theta)
    theta_new = np.zeros_like(theta)
    n = len(z)
    losses = [loss_primal(theta_tmp,z,lamb)]
    iterations = 0
    iterations_step = [0]
    steps = [theta]
    while True:
        grad = gradient_primal(theta_tmp,z)
        t, i = backtracking_primal(theta_tmp,grad,z,lamb,beta)
        iterations = iterations + 1
        theta_new = prox_opt(theta_tmp - t * grad,lamb)
        steps.append(theta_new)
        losses.append(loss_primal(theta_new,z,lamb))
        iterations = iterations + i
        iterations_step.append(iterations)
        if np.abs(losses[-1]-losses[-2]) < tol or iterations >= maxiter: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            theta_tmp = np.copy(theta_new)
    return theta_new, iterations_step, steps#, losses

def main():
    #set parameters
    data = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    n = len(data)
    y = np.copy(data)
    y[y==0] = -1
    tau = 1e3
    D = np.zeros((n-1,n))
    D[range(n-1),range(n-1)] = 1 
    D[range(n-1),range(1,n)] = -1
    lamb = 20
    delta = 0.00001
    mu = 10
    eps = 1e-8
    beta = 0.8
    m = 4 * len(y) - 2
    print(m/tau)
    #embed()
    theta = np.zeros_like(y)
    #theta = np.random.rand(len(data))
    plt.rcParams.update({'font.size'   : 22})
    u = returnFeasible(y,D,lamb,delta)
    theta = -y * np.log(np.dot(np.transpose(D),u)/(y-np.dot(np.transpose(D),u)))
    #time_newton = []
    #time_prox = []
    #for i in list(range(5,len(u),100)) + [1999]:
    #    sum = 0
    #    for j in range(10):
    #        start = time.time()
    #        u_step = u[:i]
    #        grad = gradient(D[:i,:i],u[:i],y[:i],tau,lamb)
    #        hessinv = np.linalg.inv(hessian(D[:i,:i], u[:i], y[:i], tau, lamb))
    #        t = backtracking(D[:i,:i],u[:i],y[:i],grad,hessinv,tau,lamb,beta)
    #        u_new = u[:i] - t * np.dot(hessinv,grad)
    #        end = time.time()
    #        sum = sum + (end - start)
    #    time_newton.append(sum/10)
    #    sum = 0
    #    for j in range(10):
    #        start = time.time()
    #        grad = gradient_primal(theta[:i],data[:i])
    #        t, i = backtracking_primal(theta[:i],grad,data[:i],lamb,beta)
    #        theta_new = prox_opt(theta[:i] - t * grad,lamb)
    #        end = time.time()
    #        sum = sum + (end - start)
    #    time_prox.append(sum/10)
    #plt.plot(range(len(time_newton)),time_newton,label="Newton")
    #plt.plot(range(len(time_prox)),time_prox, label="Proximal")
    #plt.xlabel("Number of elements")
    #plt.ylabel("Seconds")
    #plt.legend()
    #plt.show()
    #u_star, iterations_step_u, u_steps = centering(u,y,D,lamb=lamb,maxiter=1000,tol=1e-6)
    #print(iterations_step_u)
    #theta_star = -y * np.log(np.dot(np.transpose(D),u_star)/(y-np.dot(np.transpose(D),u_star)))
    #p_star = np.exp(theta_star)/(1+np.exp(theta_star))
    #plt.plot(range(len(p_star)),p_star,'o')
    #plt.show()

    #u_star, iterations_step_u, u_steps = centering(u,y,D,lamb=lamb,maxiter=100,tol=0)
    #theta_star, iterations_step_theta, theta_steps = training_primal(theta,data,lamb=lamb,tol=0,maxiter=100)
    #u_theta_steps = [-y * np.log(np.dot(np.transpose(D),i)/(y-np.dot(np.transpose(D),i))) for i in u_steps]
    #prox_loss = [loss_primal(i,data,lamb) for i in theta_steps]
    #barr_loss = [loss_primal(i,data,lamb) for i in u_theta_steps]
    #f_star = min(np.min(prox_loss),np.min(barr_loss)) - 1e-6
    #plt.plot(iterations_step_theta, np.array(prox_loss) - f_star, label="proximal gradient descent")
    #plt.plot(iterations_step_u, np.array(barr_loss) - f_star, label="barrier method")
    #plt.legend()
    #plt.xlabel("Iterations")
    #plt.ylabel("Criterion Gap (log scaled)")
    #plt.yscale("log")
    #plt.show()

    #4a
    u_star, iterations_u_star, steps_u_star = barrier_method(u,y,D,tau0=tau,mu=mu,eps=eps,lamb=lamb,maxiter=100)
    theta_star, iterations_theta_star, steps_iterations_theta = training_primal(theta,data)
    u_newton, iterations_u_newton, steps_u_newton = centering(u,y,D)
    print(iterations_u_newton)
    theta_u_star = -y * np.log(np.dot(np.transpose(D),u_star)/(y-np.dot(np.transpose(D),u_star)))
    theta_u_newton = -y * np.log(np.dot(np.transpose(D),u_newton)/(y-np.dot(np.transpose(D),u_newton)))
    p_u_star = np.exp(theta_u_star)/(1+np.exp(theta_u_star))
    p_u_newton = np.exp(theta_u_newton)/(1+np.exp(theta_u_newton))
    p_theta_star = np.exp(theta_star)/(1+np.exp(theta_star))
    plt.plot(range(len(data)), data, "o")
    plt.plot(range(len(data)), p_u_star, "o",label="Dual Barrier Method")
    plt.legend()
    plt.show()
    plt.plot(range(len(data)), data, "o")
    plt.plot(range(len(data)), p_u_newton, "o", label="Newton Method")
    plt.legend()
    plt.show()
    plt.plot(range(len(data)), data, "o")
    plt.plot(range(len(data)), p_theta_star, "o", label="Proximal Gradient Descent")
    plt.legend()
    plt.show()
    #4b
    #lambdas = np.logspace(np.log(0.001)/np.log(10),np.log(200)/np.log(10),num=80)
    #k = np.exp((np.log(200)-np.log(0.001))/79)
    #lambdas = np.array([0.001 * np.power(k,i) for i in range(80)])
    #lambdas = lambdas[::-1]
    #embed()
    #theta_int = np.zeros_like(theta)
    #iterations_cold = []
    #thetas = []
    #for i in lambdas:
    #    theta_curr, ii, _ = training_primal(theta_int, data,lamb=i, maxiter=1000)
    #    thetas.append(np.copy(theta_curr))
    #    iterations_cold.append(ii[-1])
    #with open("proximal_cold.pkl","wb") as file:
    #    save_theta = {"thetas":thetas,"iterations":iterations_cold}
    #    pickle.dump(save_theta,file)
    #theta_curr = theta_int
    #iterations_warm = []
    #thetas = []
    #for i in lambdas:
    #    theta_curr, ii, _ = training_primal(theta_curr, data, lamb=i, maxiter=1000)
    #    iterations_warm.append(ii[-1])
    #    thetas.append(np.copy(theta_curr))
    #with open("proximal_warm.pkl","wb") as file:
    #    save_theta = {"thetas":thetas,"iterations":iterations_warm}
    #    pickle.dump(save_theta,file)
    #plt.plot(lambdas,iterations_warm,label="Warm")
    #plt.plot(lambdas,iterations_cold,label="Cold")
    #plt.xlabel("Lambda (log scale)")
    #plt.ylabel("Number of iterations")
    #plt.legend()
    #plt.xscale("log",basex=10)
    #pylab.show()
    #lambdas = lambdas[::-1]
    #if os.path.exists("u_init.npy"): 
    #    u_init = np.load("u_init.npy")
    #else:
    #    u_init = returnFeasible(y,D,lambdas[0],delta)
    #    np.save("u_init.npy",u_init)
    #iterations_cold = []
    #us = []
    #j = 0
    #for i in lambdas:
    #    j = j + 1
    #    print("Iteration {}".format(j))
    #    u_curr, ii, _ = barrier_method(u_init, y,D,lamb=i,maxiter=100)
    #    us.append(np.copy(u_curr))
    #    print("Took {}".format(ii))
    #    iterations_cold.append(ii)
    #with open("dual_barrier_cold.pkl","wb") as file:
    #    save_u = {"us":us,"iterations":iterations_cold}
    #    pickle.dump(save_u,file)
    #u_curr = u_init
    #iterations_warm = []
    #us = []
    #j = 0
    #for i in lambdas:
    #    j = j + 1
    #    print("Iteration for {}".format(j))
    #    u_curr, ii, _ = barrier_method(u_curr, y,D,lamb=i,maxiter=100)
    #    us.append(np.copy(u_curr))
    #    print("Took {}".format(ii))
    #    iterations_warm.append(ii)
    #with open("dual_barrier_warm.pkl","wb") as file:
    #    save_u = {"us":us,"iterations":iterations_warm}
    #    pickle.dump(save_u,file)
    #plt.plot(lambdas,iterations_warm,label="Warm")
    #plt.plot(lambdas,iterations_cold,label="Cold")
    #plt.xlabel("Lambda (log scale)")
    #plt.ylabel("Number of iterations")
    #plt.legend()
    #plt.xscale("log",basex=10)
    #plt.show()

main()
