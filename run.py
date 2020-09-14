# Integrated neural network
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
    import tensorflow as tf
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.layers import Dense, Conv2D, Flatten


    class CNNModel(Model):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                             activation='relu')
            self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(512, activation="relu")
            self.d2 = Dense(1)  # C
            self.d3 = Dense(a_num, activation='softmax')  # A
            self.call(np.random.random((epochs, 84, 84, k)).astype(np.float32))

        @tf.function
        def call(self, inputs):
            x = inputs / 255.0
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            x = self.flatten(x)
            x = self.d1(x)
            ap = self.d3(x)
            v = self.d2(x)
            return ap, v


    model = CNNModel()

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

    ap, v = model(current_state)
    ap = ap.numpy()
    # v = v.numpy()

    # print(ap[0])
    assert ap.shape == (size - 1, a_num)
    assert v.shape == (size - 1, 1)

    # scattering action
    a = [0] + [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
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

            ap, v = model(current_state)
            ap = ap.numpy()
            assert ap.shape == (size - 1, a_num)
            assert v.shape == (size - 1, 1)

            # scattering action
            a = [0] + [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
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

        ap, v = model(current_state)  # dont need ap
        assert ap.shape == (size - 1, a_num)
        assert v.shape == (size - 1, 1)

        # TODO: use all information update network

        # move last one to first one
        total_state[0, :, :, :, :] = current_state

        ap, v = model(current_state)
        ap = ap.numpy()
        assert ap.shape == (size - 1, a_num)
        assert v.shape == (size - 1, 1)

        # scattering action
        a = [0] + [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
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

            # send other information
            r = comm.gather(r, root=0)
            done = comm.gather(done, root=0)
            info = comm.gather(info, root=0)



            state = state_

        # TODO: test uint8 and float32 who faster