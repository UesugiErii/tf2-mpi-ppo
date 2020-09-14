# prototype.py is used to simulate the process of sending and receiving data between brain and env
# use to make sure that the process of sending and receiving data is right
# mpirun -n 9 --oversubscribe python3 run.py

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# use to pass state
send_state_buf = np.empty((84, 84, 1), dtype=np.float32)
recv_state_buf = None

# placeholder
r = 0  # reward
done = 0  # done
info = [0]  # info
a = 0  # action

# Hyperparameter
epochs = 128  # horizon
k = 4

# brain
if rank == 0:
    recv_state_buf = np.empty((size, 84, 84, k), dtype=np.float32)
    for epoch in range(epochs):
        # recv state
        comm.Gather(send_state_buf, recv_state_buf, root=0)
        for i in range(1, size):
            assert recv_state_buf[i, 0, 0, 0] == i + epoch

        # scattering action
        a = [i + epoch for i in range(size)]
        a = comm.scatter(a, root=0)
        assert a == 0 + epoch

        # recv other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)

        assert r == [0] + [epoch + i for i in range(1, size)]
        assert done == [0] + [epoch + i for i in range(1, size)]
        assert info == [[0]] + [[epoch + i] for i in range(1, size)]

        # reset other information
        r = 0
        done = 0
        info = [0]
# env
else:
    for epoch in range(epochs):
        # fake step()
        state = np.zeros((84, 84, k), dtype=np.float32) + rank + epoch
        r = rank + epoch
        done = rank + epoch
        info = [rank + epoch]

        # send state
        send_state_buf = state
        assert id(send_state_buf) == id(state)  # dont copy
        comm.Gather(send_state_buf, recv_state_buf, root=0)

        # get action
        a = comm.scatter(a, root=0)
        assert a == rank + epoch

        # send other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)
        assert r is None and done is None and info is None