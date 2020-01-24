import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

def lassoLoss(Q,b,x,lamb):
    return np.dot(x,np.dot(Q,x)) - np.dot(b,x) + lamb * np.sum(np.abs(x))

def softThreshold(x,lamb):
    return np.sign(x) * (np.abs(x) - lamb).clip(0)

#Quadratic + Lasso solver, we develope this in the first question
def lassoSolver(Q,b,x,lamb):
    p = Q.shape[0]
    iis = np.arange(p)
    tmp = np.copy(x)
    loss = [lassoLoss(Q,b,tmp,lamb)]
    while True:
        for j in iis:
            step_iis = np.delete(iis,j)
            tmp[j] = softThreshold(b[j] - np.dot(Q[step_iis,j], tmp[step_iis]), lamb)/Q[j,j]
        loss.append(lassoLoss(Q,b,tmp,lamb))
        if loss[-1] == loss[-2]:
            break
    return tmp, loss

def primalObjective(Theta, S, lamb):
    return - np.log(np.linalg.det(Theta)) + np.trace(np.dot(S,Theta)) + lamb * np.sum(np.abs(Theta))

def dualObjective(W):
    p = W.shape[0]
    return -np.log(np.linalg.det(W)) - p

def glasso(S,lamb,t=0.001):
    #INIT
    p = S.shape[0]
    W = S + lamb * np.eye(p)
    W_prev = np.copy(W)
    #Theta = np.zeros_like(W)
    Theta = np.linalg.inv(W)
    off_diagonal_mask = np.where(~np.eye(p,dtype=bool))
    conv_threshold = t * np.average(np.abs(S[off_diagonal_mask]))
    iis = np.arange(p)
    beta_hat = np.zeros(p-1)
    beta = np.zeros(p-1)
    primalLoss = [primalObjective(Theta, S, lamb)]
    dualLoss = [dualObjective(W)]
    done = False
    while True:
        for j in iis:
            step_iis = np.delete(iis,j)
            W11_ii = np.meshgrid(step_iis,step_iis)
            W11_ii = (W11_ii[0],W11_ii[1])
            W11 = W[W11_ii]
            s12 = S[step_iis,j]
            beta_hat, _ = lassoSolver(W11,-s12,beta_hat,lamb)
            #W update
            w12 = -np.dot(W11, beta_hat)
            W[j,step_iis] = w12
            W[step_iis,j] = w12
            #Calculate Theta
            w22 = W[j,j]
            theta22 = 1/(w22 + np.dot(w12,beta_hat))
            theta12 = theta22 * beta_hat
            Theta[j,step_iis] = theta12
            Theta[step_iis,j] = theta12
            Theta[j,j] = theta22
            #Append the new loss values for primal and dual objectives
            primalLoss.append(primalObjective(Theta, S, lamb))
            dualLoss.append(dualObjective(W))
            if np.abs(dualLoss[-2]-dualLoss[-1]) < 0.00001:
                done = True
                break
            #if np.average(np.abs(W - W_prev)) < conv_threshold:
            #    done = True
            #    break
            else:
                W_prev = np.copy(W)
        if done:
            break
    return W, Theta, primalLoss, dualLoss
            
    #Calculating Theta
    #for j in iis:
    #    beta_hat = Beta_hat[:,j]
    #    step_iis = np.delete(iis,j)
    #    w12 = W[step_iis,j]
    #    w22 = W[j,j]
    #    theta22 = 1/(w22 - np.dot(w12,beta_hat))
    #    Theta[step_iis,step_iis]


def main():
    #initiliaze parameter
    plt.rcParams.update({'font.size':36})
    p = 40
    n = 100
    cov = np.random.rand(p,p)
    cov = np.dot(cov.T, cov)
    data = np.random.multivariate_normal(np.zeros(p),cov,n)
    ave = np.average(data,axis=0)
    data_mean0 = data - ave.reshape(1,40)
    S = np.dot(data_mean0.T, data_mean0)/n
    lamb = 0.6
    #Run glasso
    out = glasso(S,lamb)
    #
    primal_loss = out[2]
    dual_loss = out[3]
    change_primal = -1 * np.diff(primal_loss)
    change_dual = -1 * np.diff(dual_loss)
    #check if the algorithm is a proper descent algorihtm for either the primal or the dual problem.
    if np.all(change_primal > 0):
        print("A proper descent algorithm for the primal problem.")
    else:
        print("Not a proper descent algorithm for the primal problem.")
    if np.all(change_dual > 0):
        print("A proper descent algorithm for the dual problem.")
    else:
        print("Not a proper descent algorithm for the dual problem.")
    #plots
    plt.plot(range(len(primal_loss)),primal_loss,label="primal")
    plt.plot(range(len(dual_loss)),-1 * np.array(dual_loss),label="dual")
    plt.xlabel("iterations")
    plt.ylabel("Objective Function")
    plt.legend()
    plt.show()
    plt.bar(range(len(change_primal)),change_primal,label="primal")
    plt.xlabel("Iterations")
    plt.ylabel("Change in Objective")
    plt.legend()
    plt.show()
    plt.bar(range(len(change_dual)),change_dual,label="dual")
    plt.xlabel("Iterations")
    plt.ylabel("Change in Objective")
    plt.legend()
    plt.show()

main()
