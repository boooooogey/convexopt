from prox import prox_dp
import numpy as np
import sys
from math import inf
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from IPython import embed

def f(x,u,t,lamb):
    if np.any(x <= 0) or np.any((1-x) <= 0) or np.any((lamb-u) <= 0) or np.any((lamb+u) <= 0):
        return inf
    else:
        return np.dot(x,np.log(x)) + np.dot(1-x, np.log(1-x)) - 1/t * np.sum(np.log(x) + np.log(1-x)) - 1/t * np.sum(np.log(lamb-u)+np.log(lamb+u))

def loss(u,D,y,lamb,t):
    yDTu =  y * np.dot(np.transpose(D),u)
    return f(yDTu,u,t,lamb)

def constraints(x,D,y,lamb,t):
    u = x[:len(y)-1]
    yDTu = y * np.dot(np.transpose(D),u)
    u_lamb = u - lamb
    _u_lamb = -u -lamb
    return np.concatenate((yDTu-1,-yDTu,u_lamb,-u-lamb))

def gradient(D,u,y,t,lamb):
    yDTu = y * np.dot(np.transpose(D),u)
    fprime = np.log(yDTu)  - np.log( 1 - yDTu) - 1/t * 1/yDTu + 1/t * 1/(1-yDTu)
    gprime = y.reshape(-1,len(y)) * D
    grad_part = np.dot(gprime,fprime)
    return grad_part - 1/t * (-1/(lamb-u) + 1/(lamb+u))

def gradientFs(D,y):
    grad = np.transpose(y.reshape(-1,len(y)) * D)
    return np.concatenate((grad, -grad, np.eye(grad.shape[1]), -np.eye(grad.shape[1])))

def hessian(D,u,y,t,lamb):
    yDTu =  y * np.dot(np.transpose(D),u)
    fprime =  1/yDTu + 1/(1-yDTu)
    hessian = np.dot(fprime.reshape(1,-1) * D, np.transpose(D))
    return hessian

def hessianpd(D,x,y,t,lamb):
    u = x[:len(y)-1]
    v = x[len(y)-1:]
    block11 = hessian(D,u,y,t,lamb)
    gradientCons = gradientFs(D,y)
    block12 = np.transpose(gradientCons)
    block21 = v.reshape(-1,1) * gradientCons
    block22 = np.diag(constraints(x,D,y,lamb,t))
    return np.bmat([[block11,block12],[block21,block22]])

def rdual(x,D,y,t,lamb):
    u = x[:len(y)-1]
    v = x[len(y)-1:]
    gradientCons = gradientFs(D,y)
    return gradient(D,u,y,t,lamb) + np.dot(np.transpose(gradientCons),v)

def rcent(x,D,y,t,lamb):
    u = x[:len(y)-1]
    v = x[len(y)-1:]
    return - constraints(x,D,y,lamb,t) * v - 1/t

def r(x,D,y,lamb,t):
    return np.concatenate((rdual(x,D,y,t,lamb),rcent(x,D,y,t,lamb)))

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

def surrogate_duality_gap(x,D,y,lamb,t):
    v = x[len(y)-1:]
    cons = constraints(x,D,y,lamb,t)
    return - np.dot(cons,v)

def backtracking_dualprimal(x,step,D,y,lamb,t,):
    u = x[:len(y)-1]
    v = x[len(y)-1:]
    stepu = step[:len(y)-1]
    stepv = step[len(y)-1:]
    ii = np.where(stepv < 0)
    smax = min(1,np.min(-v[ii]/stepv[ii]))
    sloop = 0.99 * smax
    while True:
        xstep = np.copy(x)
        xstep[:len(y)-1] = u + sloop * stepu 
        if np.all(constraints(xstep,D,y,lamb,t) < 0):
            break
        else:
            sloop = sloop * beta
    rcurr = r(x,D,y,lamb,t)
    while True:
        rupdated = r(x + sloop*step,D,y,lamb,t)
        if sqrt(rupdated.dot(rupdated)) <= (1-alpha*sloop)*sqrt(rcurr.dot(rcurr)):
            break
        else:
            sloop = sloop*beta
    return sloop

def backtracking(D,u,y,tau,lamb,beta):
    t=1
    grad = gradient(D, u, y, tau, lamb)
    hessinv = np.linalg.inv(hessian(D, u, y, tau, lamb))
    v = -np.dot(hessinv,grad)
    norm2grad = np.dot(grad,v)
    currLoss = loss(u,D,y,lamb,tau)
    iterations = 0
    while True:
        iterations = iterations + 1
        lhs = loss(u+t*v,D,y,lamb,tau)
        rhs = currLoss + t/2 * norm2grad
        if lhs <= rhs:
            break
        else:
            t = beta * t
    return t, iterations

def barrier_method(u,y,D,tau0=5,mu=10,eps=1e-8,lamb=20):
    tau_step = tau0
    u_step = u
    m = 4 * len(y) - 2
    iterations = 0
    while True:
        curr_center = centering(u_step,y,D,tau=tau_step,lamb=lamb)
        u_step = curr_center[0]
        iterations = iterations + curr_center[3][-1]
        if m/tau_step < eps:
            break
        else:
            tau_step = tau_step * mu
    return u_step, iterations
        
def centering(u,y,D,beta=0.8,lamb=20,tau=1e3,tol=1e-6,maxiter=50000):
    u_new = np.zeros_like(u)
    n = len(y)
    losses = [loss(u,D,y,lamb,tau)]
    iterations_step = [0] 
    iterations = 0
    steps = [u]
    while True:
        iterations = iterations + 1
        grad = gradient(D,u,y,tau,lamb)
        hessinv = np.linalg.inv(hessian(D, u, y, tau, lamb))
        t,i = backtracking(D,u,y,tau,lamb,beta)
        u_new = u - t * np.dot(hessinv,grad)
        steps.append(u_new)
        #iterations = iterations + i
        iterations_step.append(iterations)
        losses.append(loss(u_new,D,y,lamb,tau))
        if np.abs(losses[-1]-losses[-2]) <= tol or iterations >= maxiter: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            u = u_new
    return u, losses, iterations_step, steps

def loss_primal(theta,z,lamb):
    return -np.dot(z,theta) + np.sum(np.log(1+np.exp(theta))) + lamb * np.sum(np.abs(theta[1:] - theta[:-1]))

def loss_primal_g(theta,z):
    return -np.dot(z,theta) + np.dot(theta,np.log(1+np.exp(theta)))

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
        iterations = iterations + 1
        grad = gradient_primal(theta_tmp,z)
        t, i = backtracking_primal(theta_tmp,grad,z,lamb,beta)
        theta_new = prox_opt(theta_tmp - t * grad,lamb)
        steps.append(theta_new)
        losses.append(loss_primal(theta_new,z,lamb))
        #iterations = iterations + i
        iterations_step.append(iterations)
        if np.abs(losses[-1]-losses[-2]) <= tol or iterations >= maxiter: #np.sqrt(np.dot(theta_new - theta_tmp, theta_new -theta_tmp)) < tol:
            break
        else:
            theta_tmp = np.copy(theta_new)
    return theta_new, losses, iterations_step, steps

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
    delta = 0
    mu = 10
    eps = 1e-8
    u = returnFeasible(y,D,lamb,delta)
    #embed()
    m = 4*len(y)-2
    x = np.empty(n-1 + m)
    x[:(n-1)] = u
    #theta = np.random.rand(len(data))
    plt.rcParams.update({'font.size'   : 22})
    embed()
    #training
    #theta_star = -y * np.log(np.dot(np.transpose(D),u_star)/(y-np.dot(np.transpose(D),u_star)))
    #p = np.exp(theta_star)/(1+np.exp(theta_star))
    #plt.plot(range(len(data)), data, "o")
    #plt.plot(range(len(data)), p, "o")
    #plt.show()

main()
