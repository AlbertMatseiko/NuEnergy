import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import tensorflow as tf
from launcher import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


EXPERIMENT_NAME = "SmallRNN2"

# starting training process with configs described in yml_configs directory
history = launch_exp(EXPERIMENT_NAME)
print(history.history, file=open(f"experiments/{EXPERIMENT_NAME}/history.txt", "w"))
