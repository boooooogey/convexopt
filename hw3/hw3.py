from prox import prox_dp
import numpy as np
import sys
from matplotlib import pyplot as plt

def loss_primal(theta,z,lamb):
    return -np.dot(z,theta) + np.sum(np.log(1+np.exp(theta))) + lamb * np.sum(np.abs(theta[:-1] - theta[1:]))

def loss_primal_g(theta,z):
    return -np.dot(z,theta) + np.sum(np.log(1+np.exp(theta)))

def gradient_primal(theta,z):
    return -z + np.exp(theta)/(1+np.exp(theta))

def prox_opt(theta,lamb):
    tmp = np.zeros_like(theta)
    prox_dp(len(theta),theta,lamb,tmp)
    return tmp 

def generalized_gradient(theta,grad,t,lamb):
    return (theta - prox_opt(theta-t*grad,lamb))/t

def backtracking_primal(curr,grad,z,lamb,beta):
    t=1
    iterations = 0
    while True:
        iterations = iterations + 1
        G_curr = generalized_gradient(curr,grad,t,lamb)
        rhs = loss_primal_g(curr,z) - t * np.dot(grad,G_curr) + t/2 * np.dot(G_curr,G_curr)
        lhs = loss_primal_g(curr - t*G_curr,z)
        #print(lhs, rhs)
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t, iterations 

def training_primal(theta,z,beta=0.8,lamb=20,t=1,tol=1e-6):
    theta_tmp = np.copy(theta)
    theta_new = np.zeros_like(theta)
    n = len(z)
    losses = [loss_primal(theta_tmp,z,lamb)]
    iterations = 0
    while True:
        iterations  = iterations + 1
        grad = gradient_primal(theta_tmp,z)
        t, i = backtracking_primal(theta_tmp,grad,z,lamb,beta)
        iterations = iterations + i
        theta_new = prox_opt(theta_tmp - t * grad,lamb)
        losses.append(loss_primal(theta_new,z,lamb))
        if np.abs(losses[-1]-losses[-2]) <= tol: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            theta_tmp = np.copy(theta_new)
    return theta_new, losses, iterations 

def main():
    data = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    theta = np.random.rand(len(data))
    theta = np.zeros_like(data)
    theta_new, losses, iterations = training_primal(theta,data,tol=1e-6,lamb=200)
    print(iterations)
    print(losses[-1])
    plt.plot(range(len(losses)),np.asarray(losses))
    for i in range(1,len(losses)):
        if losses[i] > losses[i-1]:
            print("i was right")
            print(i)
    plt.show()
    plt.plot(range(len(data)),data[:],"o")
    plt.plot(range(len(theta_new)),np.exp(theta_new)/(1+np.exp(theta_new)),"o")
    plt.show()

main()
