# tensorboard, save weight, exit, LearningRateSchedule, every_episode_reward
# mpirun -n 9 --oversubscribe --allow-run-as-root python3 run.py

from mpi4py import MPI
import numpy as np
from datetime import datetime

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

# parameter
logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
weight_dir = "./logs/weight/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
learning_rate = 0.001
total_step = 10 ** 7
beta = 0.01
VFcoeff = 1
clip_epsilon = 0.2
epochs = 128  # horizon
k = 4
env_name = 'PongDeterministic-v4'  # env name
import gym

env = gym.make(env_name)
a_num = env.action_space.n  # env.action_space
del env
gamma = 0.99  # discount reward

# brain
if rank == 0:
    import numba


    @numba.jit(nopython=True)
    def calc_real_v_and_adv_GAE(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        adv = np.zeros((length + 1, num), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            delta = r[t, :] + v[t + 1, :] * gamma * (1 - done[t, :]) - v[t, :]
            adv[t, :] = delta + gamma * 0.95 * adv[t + 1, :] * (1 - done[t, :])

        adv = adv[:-1, :]

        realv = adv + v[:-1, :]

        return realv, adv


    @numba.jit(nopython=True)
    def calc_real_v_and_adv(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        realv = np.zeros((length + 1, num), dtype=np.float32)
        adv = np.zeros((length, num), dtype=np.float32)

        realv[-1, :] = v[-1, :] * (1 - done[-1, :])

        for t in range(length - 1, -1, -1):
            realv[t, :] = realv[t + 1, :] * gamma * (1 - done[t, :]) + r[t, :]
            adv[t, :] = realv[t, :] - v[t, :]

        return realv[:-1, :], adv  # end_v dont need


    import tensorflow as tf
    import os

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
    import tensorflow.keras.optimizers as optim

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, init_lr):
            super(CustomSchedule, self).__init__()
            self.lr = init_lr
            self.max_learning_times = total_step * 3 // epochs // (size - 1)

        def __call__(self, step):
            # step start from 0
            # every time call , step automatic += 1
            self.last_lr = self.lr * ((self.max_learning_times - step) / self.max_learning_times)
            return self.last_lr

        def get_config(self):
            return self.last_lr


    # optimizer = optim.Adam(learning_rate=CustomSchedule(learning_rate))  # linearly annealed
    optimizer = optim.Adam(learning_rate=0.001)  # no annealed

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

        @tf.function
        def loss(self, state, a, adv, real_v, old_ap):
            res = self.call(state)
            error = res[1][:, 0] - real_v
            L = tf.reduce_sum(tf.square(error))

            adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
            batch_size = state.shape[0]
            all_act_prob = res[0]
            selected_prob = tf.reduce_sum(a * all_act_prob, axis=1)
            old_prob = tf.reduce_sum(a * old_ap, axis=1)

            r = selected_prob / (old_prob + 1e-6)

            H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

            Lclip = tf.reduce_sum(
                tf.minimum(
                    tf.multiply(r, adv),
                    tf.multiply(
                        tf.clip_by_value(
                            r,
                            1 - clip_epsilon,
                            1 + clip_epsilon
                        ),
                        adv
                    )
                )
            )

            return -(Lclip - VFcoeff * L + beta * H) / batch_size

        @tf.function
        def train(self, batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap):
            with tf.GradientTape() as tape:
                loss_value = self.loss(batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap)

            grads = tape.gradient(loss_value, self.trainable_weights)
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            return loss_value


    model = CNNModel()

    total_state = np.empty((epochs, size - 1, 84, 84, k), dtype=np.float32)
    total_v = np.empty((epochs + 1, size - 1), dtype=np.float32)
    total_a = np.empty((epochs, size - 1), dtype=np.int32)
    total_r = np.zeros((epochs, size - 1), dtype=np.float32)
    total_done = np.zeros((epochs, size - 1), dtype=np.float32)
    total_old_ap = np.zeros((epochs, size - 1, a_num), dtype=np.float32)  # old action probability
    recv_state_buf = np.empty((size, 84, 84, k), dtype=np.float32)  # use to recv data

    learning_step = 0
    remain_step = total_step // (size - 1)

    all_reward = np.zeros((size - 1,), dtype=np.float32)  # Used to record the reward of each episode
    one_episode_reward_index = 0  # all env episode index

    # loop
    # ↓ <-------- ↑
    # 1 -> 255 -> 1

    # first one
    comm.Gather(send_state_buf, recv_state_buf, root=0)
    remain_step -= 1
    current_state = recv_state_buf[1:size, :, :, :]
    total_state[0, :, :, :, :] = current_state

    ap, v = model(current_state)
    ap = ap.numpy()
    v = v.numpy()
    v.resize((size - 1,))

    # scattering action
    a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
    total_a[0, :] = a
    a = [0] + a
    a = comm.scatter(a, root=0)

    # recv other information
    r = comm.gather(r, root=0)
    done = comm.gather(done, root=0)
    info = comm.gather(info, root=0)
    del r[0]
    del done[0]
    del info[0]

    total_v[0, :] = v
    total_r[0, :] = np.array(r, dtype=np.float32)
    total_done[0, :] = np.array(done, dtype=np.float32)
    all_reward += r
    for i, is_done in enumerate(done):
        if is_done:
            tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
            one_episode_reward_index += 1
            all_reward[i] = 0


    total_old_ap[0, :] = np.array(ap, dtype=np.float32)

    # reset other information
    r = 0
    done = 0
    info = 0

    # 255+1 loop
    while 1:
        for epoch in range(1, epochs):
            # recv state
            comm.Gather(send_state_buf, recv_state_buf, root=0)
            remain_step -= 1  # After every recv data minus 1
            if not remain_step:
                break  # leave for loop

            current_state = recv_state_buf[1:size, :, :, :]
            total_state[epoch, :, :, :, :] = current_state

            ap, v = model(current_state)
            ap = ap.numpy()
            v = v.numpy()
            v.resize((size - 1,))

            # scattering action
            a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
            total_a[epoch, :] = a
            a = [0] + a
            a = comm.scatter(a, root=0)

            # recv other information
            r = comm.gather(r, root=0)
            done = comm.gather(done, root=0)
            info = comm.gather(info, root=0)
            del r[0]
            del done[0]
            del info[0]

            total_v[epoch, :] = v
            total_r[epoch, :] = np.array(r, dtype=np.float32)
            total_done[epoch, :] = np.array(done, dtype=np.float32)
            all_reward += r
            for i, is_done in enumerate(done):
                if is_done:
                    tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
                    one_episode_reward_index += 1
                    all_reward[i] = 0
            total_old_ap[epoch, :] = np.array(ap, dtype=np.float32)

            # reset other information
            r = 0
            done = 0
            info = 0
        if not remain_step:
            print(rank, 'finished')
            model.save_weights(weight_dir + str(learning_step), save_format='tf')
            break  # leave while 1 loop

        # last one
        comm.Gather(send_state_buf, recv_state_buf, root=0)
        remain_step -= 1  # After every recv data minus 1
        # if now remain_step == 0, then exit after last learning

        current_state = recv_state_buf[1:size, :, :, :]

        ap, v = model(current_state)  # dont need ap
        v = v.numpy()
        v.resize((size - 1,))
        total_v[-1, :] = v

        # critic_v   advantage_v
        total_real_v, total_adv = calc_real_v_and_adv_GAE(total_v, total_r, total_done)
        total_state.resize((epochs * (size - 1), 84, 84, k))
        total_a.resize((epochs * (size - 1),))
        total_old_ap.resize((epochs * (size - 1), a_num))
        total_adv.resize((epochs * (size - 1),))
        total_real_v.resize((epochs * (size - 1),))

        print('learning' + '-' * 35 + str(learning_step) + '/' + str(total_step // epochs // (size - 1)))
        for _ in range(3):
            sample_index = np.random.choice(epochs * (size - 1), size=epochs * (size - 1) // 4)
            loss = model.train(total_state[sample_index],
                               tf.one_hot(total_a, depth=a_num).numpy()[sample_index],
                               total_adv[sample_index], total_real_v[sample_index],
                               total_old_ap[sample_index])
        learning_step += 1
        if learning_step % (total_step // epochs // (size - 1) // 200) == 0:  # recode 200 times
            tf.summary.scalar('loss', data=loss, step=learning_step)
        if learning_step % (total_step // epochs // (size - 1) // 3) == 0:  # recode 3 times
            model.save_weights(weight_dir + str(learning_step), save_format='tf')

        total_state.resize((epochs, size - 1, 84, 84, k))
        total_a.resize((epochs, size - 1))
        total_old_ap.resize((epochs, size - 1, a_num))

        # exit after last learning
        if not remain_step:
            print(rank, 'finished')
            model.save_weights(weight_dir + str(learning_step), save_format='tf')
            break  # leave while 1

        # move last one to first one
        ap, v = model(current_state)
        ap = ap.numpy()
        v = v.numpy()
        v.resize((size - 1,))

        # scattering action
        a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size - 1)]
        total_a[0, :] = a
        a = [0] + a
        a = comm.scatter(a, root=0)

        # recv other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)
        del r[0]
        del done[0]
        del info[0]

        total_v[0, :] = v
        total_r[0, :] = np.array(r, dtype=np.float32)
        total_done[0, :] = np.array(done, dtype=np.float32)
        all_reward += r
        for i, is_done in enumerate(done):
            if is_done:
                tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
                one_episode_reward_index += 1
                all_reward[i] = 0
        total_old_ap[0, :] = np.array(ap, dtype=np.float32)

        # reset other information
        r = 0
        done = 0
        info = 0


# env
else:
    from atari_wrappers import *

    env = gym.make(env_name)
    env = WarpFrame(env, width=84, height=84, grayscale=True)
    env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)

    # random init
    np.random.seed(rank)
    env.seed(rank)

    state = np.array(env.reset(), dtype=np.float32)
    one_episode_reward = 0
    count_episode = 0
    remain_step = total_step // (size - 1)
    while True:
        # send state
        send_state_buf = state
        comm.Gather(send_state_buf, recv_state_buf, root=0)
        remain_step -= 1  # After every send data minus 1
        if not remain_step:
            print(rank, 'finished')
            break

        # get action
        a = comm.scatter(a, root=0)

        state_, r, done, info = env.step(a)
        one_episode_reward += r
        state_ = np.array(state_, dtype=np.float32)

        # Fix Bug: this should on send above
        if done:
            state_ = np.array(env.reset(), dtype=np.float32)
            print(rank, count_episode, one_episode_reward)
            count_episode += 1
            one_episode_reward = 0

        # send other information
        r = comm.gather(r, root=0)
        done = comm.gather(done, root=0)
        info = comm.gather(info, root=0)

        state = state_

        # TODO: test uint8 and float32 who faster