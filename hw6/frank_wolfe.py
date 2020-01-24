from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from math import isclose
import numpy as np

def svd(m,**kwargs):
    nc = kwargs.get('n_components', 1)
    ni = kwargs.get('n_iter', 10) 
    rs = kwargs.get('random_state', 42)
    return randomized_svd(m, n_components=nc, n_iter=ni, random_state=rs)

def loss(B,Y,mask):
    return 0.5 * np.sum(np.power(B[mask] - Y[mask],2))

def gradient(B,Y,mask):
    delta_B = (B-Y)
    delta_B[np.logical_not(mask)] = 0
    return delta_B

def general_gradient(B_k_1,Y,mask,t,t_k):
    return (B_k_1 - projection(B_k_1 - t_k * gradient(B_k_1, Y, mask),t)) /t_k

def frobenius_norm(m):
    return np.sum(np.power(m,2))

def trace_norm(m):
    r,c = m.shape
    #print("trace norm dimension: {}.".format(min(r,c)))
    _, s, _ = svd(m, n_components=min(r,c))
    #print(s)
    return np.sum(s)

def subgradient_op(B,Y,mask,t):
    u, _, vt = svds(gradient(B,Y,mask),k=1)
    sub = -t * u @ vt
    return sub

def frank_wolfe_iteration(B,Y,gamma,S):
    B_next = B - gamma * (B - S)
    return B_next

def frank_wolfe(B,Y,mask,t,use_backtracking=False,tol=0,max_iter=15000):
    B_k_1 = B
    losses = [loss(B, Y, mask)]
    #print(trace_norm(B_k_1))
    k = 0
    while True:
        k += 1
        S = subgradient_op(B_k_1,Y,mask,t)
        grad = gradient(B_k_1,Y,mask)
        if use_backtracking:
            gamma = frank_wolfe_backtracking(B_k_1,Y,mask,S,grad)
        else:
            gamma = 2/(k+1) 
        B_k = frank_wolfe_iteration(B_k_1,Y,gamma,S)
        #print(trace_norm(B_k))
        losses.append(loss(B_k,Y,mask))
        if np.trace(grad.T @ (B_k_1 - S)) <= tol or k >= max_iter:
            break
        else:
            B_k_1 = B_k
    return B_k, np.array(losses)

def frank_wolfe_backtracking(B,Y,mask,S,grad,beta=0.8):
    delta = B - S
    t_k = 1
    while loss(B - t_k * delta, Y, mask) > loss(B,Y,mask) - 0.5 * t_k * np.trace(grad.T @ delta):
        t_k = beta * t_k
    return t_k

def projection(B,t,number_of_components=1):
    u, s, vt = svds(B,k=number_of_components)
    n = min(B.shape[0],B.shape[1])
    B_tmp = B - u @ np.diag(s) @ vt
    theta = (np.sum(s) - t)
    for i in range(2,n+1):
        u_tmp, s_tmp, vt_tmp = svds(B_tmp,k=number_of_components)
        if  theta >= s_tmp[-1]:
            break
        else:
            B_tmp = B_tmp - u_tmp @ np.diag(s_tmp) @ vt_tmp
            u = np.concatenate((u,u_tmp),axis=1) 
            vt = np.concatenate((vt,vt_tmp), axis=0)
            s = np.concatenate((s,s_tmp))
            theta = (np.sum(s) - t)/len(s)
    theta = max(theta, 0)
    s = (s - theta).clip(0)
    assert np.all(s >= 0 )
    #if not np.all(s >= 0):
    #    negative_all = s < 0
    #    while not isclose(t,np.sum((s-theta).clip(0))) and np.sum((s-theta).clip(0)) > t:
    #        tmp = s-theta
    #        negative_new = np.logical_xor(tmp < 0, negative_all)
    #        positive_all = tmp > 0
    #        inc = np.sum(tmp[negative_new])
    #        theta -= inc/np.sum(positive_all)
    #        negative_all = np.logical_or(negative_new, negative_all)
    #    s = (s-theta).clip(0)
    return u @ np.diag(s) @ vt

def projected_gradient_descent(B,Y,mask,t,tol=0,use_backtracking=False,use_acceleration=False,max_iter=15000):
    B_k_1 = B
    v = B
    losses = [loss(B, Y, mask)]
    t_k = 1
    k = 0
    while True:
        k += 1
        if use_backtracking:
            t_k = backtracking(v,Y,mask,t)
            #if t_k != 1:
            #    print(t_k)
            #print(_k)
        #print(t_k)
        B_k = projection(v - t_k * gradient(v, Y, mask),t)
        if use_acceleration and k > 2:
            v = B_k + (k-2)/(k+1) * (B_k - B_k_1)
        else:
            v = B_k
        losses.append(loss(B_k,Y,mask))
        if frobenius_norm(B_k-B_k_1) <= tol or k >= max_iter:
            break
        else:
            B_k_1 = B_k
    return B_k, np.array(losses)

def backtracking(B,Y,mask,t,beta=0.8):
    t_k = 1
    g = gradient(B,Y,mask)
    while True:
        G_t = general_gradient(B,Y,mask,t,t_k)
        if loss(B-t_k*G_t,Y,mask) <= loss(B,Y,mask) - t_k * np.trace(g.T @ G_t) + 0.5 * t_k * np.trace(G_t.T @ G_t): 
            break
        else:
            t_k = t_k * beta
    return t_k
