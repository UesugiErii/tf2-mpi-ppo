# Ensure the order of the data is correct
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

# test variable, for unit test
count = 0

# brain
if rank == 0:
    total_state = np.empty((epochs + 1, size - 1, 84, 84, k), dtype=np.float32)
    total_v = np.empty((epochs + 1, size - 1), dtype=np.float32)
    total_a = np.empty((epochs, size - 1), dtype=np.float32)
    total_r = np.zeros((epochs, size - 1), dtype=np.float32)
    total_done = np.zeros((epochs, size - 1), dtype=np.float32)
    # total_old_ap = np.zeros((epochs, size - 1, ), dtype=np.float32)
    recv_state_buf = np.empty((size, 84, 84, k), dtype=np.float32)

    # loop
    # ↓ <-------- ↑
    # 1 -> 255 -> 1
    # first one
    comm.Gather(send_state_buf, recv_state_buf, root=0)
    current_state = recv_state_buf[1:size, :, :, :]
    total_state[0, :, :, :, :] = current_state

    assert current_state[2, 0, 0, 0] == 3 + count  # choose rank=3 to test

    # TODO: calc v and action prob
    # scattering action
    a = [i + count for i in range(size)]
    a = comm.scatter(a, root=0)

    # recv other information
    r = comm.gather(r, root=0)
    done = comm.gather(done, root=0)
    info = comm.gather(info, root=0)
    assert r == [0] + [count + i for i in range(1, size)]
    assert done == [0] + [count + i for i in range(1, size)]
    assert info == [[0]] + [[count + i] for i in range(1, size)]

    count += 1
    # reset other information
    r = 0
    done = 0
    info = [0]

    # 255 loop ( Don't write exit conditions temporarily )
    while 1:
        for epoch in range(epochs - 1):
            # recv state
            comm.Gather(send_state_buf, recv_state_buf, root=0)

            current_state = recv_state_buf[1:size, :, :, :]
            total_state[epoch + 1, :, :, :, :] = current_state

            assert current_state[2, 0, 0, 0] == 3 + count  # choose rank=3 to test

            # TODO: calc v and action prob

            # scattering action
            a = [i + count for i in range(size)]
            a = comm.scatter(a, root=0)

            # recv other information
            r = comm.gather(r, root=0)
            done = comm.gather(done, root=0)
            info = comm.gather(info, root=0)
            assert r == [0] + [count + i for i in range(1, size)]
            assert done == [0] + [count + i for i in range(1, size)]
            assert info == [[0]] + [[count + i] for i in range(1, size)]
            count += 1

            # reset other information
            r = 0
            done = 0
            info = [0]
        # last one
        comm.Gather(send_state_buf, recv_state_buf, root=0)

        current_state = recv_state_buf[1:size, :, :, :]
        total_state[epoch + 1, :, :, :, :] = current_state

        # TODO: calc v and action prob, only need v

        # TODO: use all information update network

        # move last one to first one
        total_state[0, :, :, :, :] = current_state

        assert current_state[2, 0, 0, 0] == 3 + count  # choose rank=3 to test
        # TODO: calc v and action prob for current_state

        # scattering action
        a = [i + count for i in range(size)]
        a = comm.scatter(a, root=0)

        # recv other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)
        assert r == [0] + [count + i for i in range(1, size)]
        assert done == [0] + [count + i for i in range(1, size)]
        assert info == [[0]] + [[count + i] for i in range(1, size)]
        count += 1

        # reset other information
        r = 0
        done = 0
        info = [0]


# env
else:
    while 1:
        # fake step()
        state = np.zeros((84, 84, k), dtype=np.float32) + rank + count
        r = rank + count
        done = rank + count
        info = [rank + count]

        # send state
        send_state_buf = state
        comm.Gather(send_state_buf, recv_state_buf, root=0)

        # get action
        a = comm.scatter(a, root=0)

        assert a == rank + count

        # send other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)

        count += 1