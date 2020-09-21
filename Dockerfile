FROM tensorflow/tensorflow:latest-gpu

WORKDIR /root

RUN apt update \
    && apt install -y libgl1-mesa-glx libopenmpi-dev wget openssh-server\
    && rm -rf /var/lib/apt/lists/* \
    && pip3 --no-cache-dir install opencv-python gym numba mpi4py \
    && pip3 --no-cache-dir install gym[atari] \
    && wget https://github.com/UesugiErii/tf2-mpi-ppo/archive/master.zip \
    && unzip master.zip \
    && rm master.zip
    
CMD /bin/bash

# mpirun -n 8 --oversubscribe --allow-run-as-root python3 /root/tf2-mpi-ppo-master/run.py
