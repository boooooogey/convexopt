# In the first Python interactive shell
import numpy as np
from multiprocessing import shared_memory
import multiprocessing as mp

def main():
    a = np.random.rand(6,6) # Start with an existing NumPy array
    x = np.random.rand(5)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    shm2 = shared_memory.SharedMemory(create=True, size=x.nbytes)
    # Now create a NumPy array backed by shared memory
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    y = np.ndarray(x.shape, dtype=x.dtype, buffer=shm2.buf)
    b[:] = a[:]  # Copy the original data into shared memory
    y[:] = x[:]
    print("main: {}".format(b))
    print("main: {}".format(y))
    name = shm.name  # We did not specify a name so one was chosen for us
    name2 = shm2.name
    print(name)
    print(name2)
    (c1,c2) = mp.Pipe()
    (c3,c4) = mp.Pipe()
    cs = [c2,c4]
    p=[]
    for i in range(2):
        p.append(mp.Process(target=node,args=(name,cs[i],a.shape,a.dtype,name2,x.shape,x.dtype)))
        p[i].start()
    print(c1.recv())
    print(c3.recv())
    del b  # Unnecessary; merely emphasizing the array is no longer used
    shm.close()
    shm.unlink()  # Free and release the shared memory block at the very end


def node(name_b,c2,shape_b,type_b,name2,xs,xd):
    # In either the same shell or a new Python shell on the same machine
    # Attach to the existing shared memory block
    existing_shm = shared_memory.SharedMemory(name=name_b)
    existing_shm2 = shared_memory.SharedMemory(name=name2)
    # Note that a.shape is (6,) and a.dtype is np.int64 in this example
    c = np.ndarray(shape_b, dtype=type_b, buffer=existing_shm.buf)
    d = np.ndarray(xs, dtype=xd, buffer=existing_shm2.buf)
    while True:
        print("node c: {}".format(c))
        print("node d: {}".format(d))
    c2.send("a") 
    del c
    existing_shm.close()

main()
