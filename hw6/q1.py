from sklearn.utils.extmath import randomized_svd
import numpy as np

def svd(m,**kwargs):
    nc = kwargs.get('n_components', 1)
    ni = kwargs.get('n_iter', 10) 
    rs = kwargs.get('random_state', None)
    return randomized_svd(m, n_components=nc, n_iter=ni, random_state=rs)

def loss(B,Y,mask):
    return np.sum(np.power(B[mask] - Y[mask],2))

def gradient(B,Y,mask):
    delta_B = 2*(B-Y)
    #delta_B[np.logical_not(mask)] = 0
    return delta_B

def frobenius_norm(m):
    return np.sum(np.power(m,2))

def trace_norm(m):
    _, s, _ = np.linalg.svd(m)
    return np.sum(np.abs(s))

def subgradient_op(B,Y,mask,t):
    u, _, vt = svd(gradient(B,Y,mask))
    sub = -t * u @ vt
    return sub

def frank_wolfe_iteration(B,Y,mask,t,gamma):
    S = subgradient_op(B,Y,mask,t)
    B_next = (1-gamma) * B + gamma * S
    return B_next

def frank_wolfe(B,Y,mask,t,tol=0):
    B_k_1 = B
    losses = [loss(B, Y, mask)]
    k = 0
    print(trace_norm(B_k_1))
    while True:
        k += 1
        gamma = 2/(k+1) 
        B_k = frank_wolfe_iteration(B_k_1,Y,mask,t,gamma)
        print(trace_norm(B_k))
        losses.append(loss(B_k,Y,mask))
        if frobenius_norm(losses[-1] - losses[-2]) <= tol:
            break
        else:
            B_k_1 = B_k
    return B_k, losses
