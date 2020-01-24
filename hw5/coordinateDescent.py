import numpy as np
from matplotlib import pyplot as plt

def lassoLoss(Q,b,x,lamb):
    return 0.5*np.dot(x,np.dot(Q,x)) - np.dot(b,x) + lamb * np.sum(np.abs(x))

def softThreshold(x,lamb):
    return np.sign(x) * (np.abs(x) - lamb).clip(0)

def gradient(Q,b,x):
    return np.dot(Q, x) - b 
def generateGeneralizedGradient(Q,b,lamb,j):
    def G(x,t):
        newx = np.copy(x)
        newx[j] = softThreshold(x[j] - t*gradient(Q,b,x), t*lamb)
        return (x-newx)/t
    return G

def g(Q,b,x):
    return 0.5 * np.dot(x,np.dot(Q,x)) - np.dot(b,x)

def backtracking(Q,b,x,lamb,tinit,beta,j):
    t = tinit
    G = generateGeneralizedGradient(Q[:,j],b[j],lamb,j)
    step = 0
    while True:
        G_val = G(x,t)
        step = step + 1
        lhs = g(Q,b,x-t*G_val)
        rhs = g(Q,b,x) - t * np.dot(gradient(Q,b,x),  G_val) + t/2 * np.dot(G_val,G_val)
        if lhs <= rhs:
            break
        else:
            t = t * beta
    return t, step

def proximalCoordinateDescent(Q,b,x,lamb,tinit,beta,thr,method,maxiterations=10000000000000):
    p = Q.shape[0]
    tmp = np.copy(x)
    loss = [lassoLoss(Q,b,tmp,lamb)]
    step = 0
    steps = [0]
    while True:
        for j in range(p):
            if method == 1: #backtracking
                t, s = backtracking(Q,b,tmp,lamb,tinit,beta,j)
                step = step + s
            if method == 2: #Exact
                t = 1/Q[j,j]
            if method == 3: #Fixed step
                t = tinit
            tmp[j] = softThreshold(tmp[j] - t*gradient(Q[:,j],b[j],tmp), t*lamb)
            step = step + 1
        loss.append(lassoLoss(Q,b,tmp,lamb))
        steps.append(step)
        if np.abs(loss[-1] - loss[-2]) <= thr or step >= maxiterations:
            break
    return tmp, np.array(loss), np.array(steps)

def plotTrajectories(data,colors,names,maxnum):
    ylim = 0
    n = len(colors)
    for i,(y,x) in enumerate(data):
        if i < n:
            plt.plot(x,y,colors[i%n],alpha=0.7,label=names[i%n])
        else:
            plt.plot(x,y,colors[i%n],alpha=0.7)
        if np.min(y[np.where(x <= maxnum)]) < ylim:
            ylim = np.min(y[np.where(x <= maxnum)])
    plt.legend()
    plt.xlim((0,maxnum))
    plt.ylim((ylim-np.abs(ylim)*0.01,np.abs(ylim)*0.01))
    plt.xlabel("iterations")
    plt.ylabel("Objective Function")
    plt.show()
        

def main():
    #initiliaze parameter
    plt.rcParams.update({'font.size':36})
    p = 100
    tol = 0#1e-6
    lamb = 0.4
    number_of_experiments = 50
    data = [] 
    for i in range(number_of_experiments):
        # Simulate data
        S = np.random.rand(p,p)
        Q = np.dot(S.T, S)  + np.eye(p)
        b = np.random.rand(p)
        x = np.zeros(p)
        # Exact
        out = proximalCoordinateDescent(Q,b,x,lamb,0.01,0.8,tol,2)
        data.append((out[1],out[2]))
        # Fixed
        out_prox = proximalCoordinateDescent(Q,b,x,lamb,0.001,0.8,tol,3,maxiterations = out[2][-1])
        data.append((out_prox[1],out_prox[2]))
        # Backtracking
        out_prox_back = proximalCoordinateDescent(Q,b,x,lamb,1,0.8,tol,1)
        data.append((out_prox_back[1],out_prox_back[2]))
    plotTrajectories(data,['b','r','g'],["Exact","Fixed(t=0.001)","Backtracking"],10000)

main()
