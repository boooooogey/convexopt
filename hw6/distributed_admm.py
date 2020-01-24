import numpy as np
import pickle as pkl
import multiprocessing as mp
import sys
from multiprocessing import shared_memory

def loss(A,b,x,lamb):
    return f(A,b,x) + g(x,lamb)

def f(A,b,x):
    return pow(np.linalg.norm(A @ x - b),2)

def g(x,lamb):
    return lamb * np.sum(np.abs(x))

def close_form_solution(C,rho_I,A_T_b,rho,z,u):
    return np.linalg.inv(C + rho_I) @ (A_T_b + rho*(z - u))

def soft_threshold(x,lamb):
    return np.sign(x) * (np.abs(x) - lamb).clip(0)

def admm_node(name_A, name_b, shape_A, shape_b, type_A, type_b, i, m, rho, con):
    shm_A = shared_memory.SharedMemory(create=False,name=name_A)
    shm_b = shared_memory.SharedMemory(create=False,name=name_b)
    A = np.ndarray(shape_A, dtype=type_A, buffer=shm_A.buf)
    b = np.ndarray(shape_b, dtype=type_b, buffer=shm_b.buf)
    n = A.shape[0]
    n_sample = int(n/m)
    if i-1 == m:
        C = A[i*n_sample:,].T @ A[i*(n_sample):,]
        A_T_b = A[i*(n_sample):,].T @ b[i*(n_sample):]
    else:
        C = A[i*(n_sample):(i+1)*(n_sample),].T @ A[i*(n_sample):(i+1)*(n_sample),]
        A_T_b = A[i*(n_sample):(i+1)*(n_sample),].T @ b[i*(n_sample):(i+1)*(n_sample)]
    s = C.shape
    rho_I = rho * np.eye(s[0])
    while True:
        z, u, end = con.recv()
        if end:
            break
        x_next = close_form_solution(C,rho_I,A_T_b,rho,z,u)
        con.send(x_next)
    del A
    del b
    shm_A.close()
    shm_b.close()
    con.send(True)

def center_node(A,b,number_of_processes,rho,lamb,tol=0,max_iteration=1000):
    n = A.shape[0]
    p = A.shape[1]
    shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
    shm_b = shared_memory.SharedMemory(create=True, size=b.nbytes)
    shared_A = np.ndarray(A.shape, dtype=A.dtype, buffer=shm_A.buf) 
    shared_b = np.ndarray(b.shape, dtype=b.dtype, buffer=shm_b.buf) 
    shared_A[:] = A[:] 
    shared_b[:] = b[:]
    name_A = shm_A.name
    name_b = shm_b.name
    m = int(n/number_of_processes)
    com = []
    processes = []
    losses = [loss(A,b,np.zeros(p),lamb)]
    k = 0
    for i in range(number_of_processes):
        (c1,c2) = mp.Pipe()
        com.append(c1)
        processes.append(mp.Process(target=admm_node,args=(name_A,name_b,A.shape,b.shape,A.dtype,b.dtype,i,number_of_processes,rho,c2)))
        processes[i].start()
    us = [np.zeros(p) for i in range(number_of_processes)]
    xs = []
    z = np.zeros(p)
    u_mean = np.zeros(p)
    while True:
        k += 1
        for (ni,i) in enumerate(com):
            i.send([z, us[ni], False])
        for (ni,i) in enumerate(com):
            xi = i.recv()
            if len(xs) < ni+1:
                xs.append(xi)
            else:
                xs[ni] = xi
        x_mean = np.mean(xs,axis=0)
        z = soft_threshold(x_mean+u_mean,lamb/(rho * number_of_processes))
        for (ni,i) in enumerate(xs):
            us[ni] = us[ni] + xs[ni] - z
        u_mean = np.mean(us,axis=0)
        losses.append(loss(A,b,z,lamb))
        if np.abs(losses[-1]-losses[-2]) <= tol or k >= max_iteration:
            for i in com:
                i.send([z, us[ni], True])
            break
    for (ni,i) in enumerate(com):
        if not i.recv():
            print("something wrong with {} node.".format(ni))
    del shared_A
    del shared_b
    shm_A.close()
    shm_A.unlink()
    shm_b.close()
    shm_b.unlink()

    for i in processes:
        i.join()

    return z, losses

def main():
    condition_numbers = [1, 5, 10]
    As = dict()
    bs = dict()
    for i in condition_numbers:
        As[i] = np.load("A_{}.npy".format(i))
        bs[i] = np.load("b_{}.npy".format(i))
    lamb = np.linspace(float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]))
    print(lamb)
    rho = np.linspace(float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6]))
    print(rho)
    number_of_processes = [2,4,6,8]
    out_admm = dict()
    for i in condition_numbers:
        for j in number_of_processes:
            for k in lamb:
                for l in rho:
                    print(i,j,k,l)
                    out_admm[(i,j,k,l)] = center_node(As[i],bs[i],j,l,k)
    with open("admm_result.pkl",'wb') as file:
        pkl.dump(out_admm,file)
    
main()
