# ensure that true env is work
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
info = 0  # info
a = 0  # action

# Hyperparameter
epochs = 128  # horizon
k = 4
env_name = 'PongDeterministic-v4'  # env name
import gym

env = gym.make(env_name)
a_num = env.action_space.n  # env.action_space
del env

# brain
if rank == 0:
    total_state = np.empty((epochs + 1, size - 1, 84, 84, k), dtype=np.float32)
    total_v = np.empty((epochs + 1, size - 1), dtype=np.float32)
    total_a = np.empty((epochs, size - 1), dtype=np.float32)
    total_r = np.zeros((epochs, size - 1), dtype=np.float32)
    total_done = np.zeros((epochs, size - 1), dtype=np.float32)
    total_old_ap = np.zeros((epochs, size - 1, a_num), dtype=np.float32)
    recv_state_buf = np.empty((size, 84, 84, k), dtype=np.float32)

    # loop
    # ↓ <-------- ↑
    # 1 -> 255 -> 1
    # first one
    comm.Gather(send_state_buf, recv_state_buf, root=0)
    current_state = recv_state_buf[1:size, :, :, :]
    total_state[0, :, :, :, :] = current_state


    # TODO: calc v and action prob
    # scattering action
    a = [0 for i in range(size)]
    a = comm.scatter(a, root=0)

    # recv other information
    r = comm.gather(r, root=0)
    done = comm.gather(done, root=0)
    info = comm.gather(info, root=0)


    # reset other information
    r = 0
    done = 0
    info = 0

    # 255 loop ( Don't write exit conditions temporarily )
    while 1:
        for epoch in range(epochs - 1):
            # recv state
            comm.Gather(send_state_buf, recv_state_buf, root=0)

            current_state = recv_state_buf[1:size, :, :, :]
            total_state[epoch + 1, :, :, :, :] = current_state


            # TODO: calc v and action prob

            # scattering action
            a = [0 for i in range(size)]
            a = comm.scatter(a, root=0)

            # recv other information
            r = comm.gather(r, root=0)
            done = comm.gather(done, root=0)
            info = comm.gather(info, root=0)



            # reset other information
            r = 0
            done = 0
            info = 0
        # last one
        comm.Gather(send_state_buf, recv_state_buf, root=0)

        current_state = recv_state_buf[1:size, :, :, :]
        total_state[epoch + 1, :, :, :, :] = current_state

        # TODO: calc v and action prob, only need v

        # TODO: use all information update network

        # move last one to first one
        total_state[0, :, :, :, :] = current_state

        # TODO: calc v and action prob for current_state

        # scattering action
        a = [0 for i in range(size)]
        a = comm.scatter(a, root=0)

        # recv other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)


        # reset other information
        r = 0
        done = 0
        info = 0


# env
else:
    from atari_wrappers import *

    while 1:
        env = gym.make(env_name)
        env = WarpFrame(env, width=84, height=84, grayscale=True)
        env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)

        # random
        np.random.seed(rank)
        env.seed(rank)

        state = np.array(env.reset(), dtype=np.float32)
        while True:
            # send state
            send_state_buf = state
            comm.Gather(send_state_buf, recv_state_buf, root=0)

            # get action
            a = comm.scatter(a, root=0)

            state_, r, done, info = env.step(a)
            state_ = np.array(state_, dtype=np.float32)

            if done:
                state_ = np.array(env.reset(), dtype=np.float32)
                print(rank, done)

            # send other information
            r = comm.gather(r, root=0)
            done = comm.gather(done, root=0)
            info = comm.gather(info, root=0)



            state = state_

        # TODO: test uint8 and float32 who faster