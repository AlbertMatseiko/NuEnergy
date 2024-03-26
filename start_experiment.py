import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
from NNBlock.launcher import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

EXPERIMENT_NAME = "25_03_BigRNN_OnFlatSpec"
DESCRIPTION = """Training BIG RNN on a file with flat spectrum of energy. WITH gauss noise. Bigger lr_rate."""

# starting training process with configs described in yml_configs directory
model, history = launch_exp(EXPERIMENT_NAME, DESCRIPTION=DESCRIPTION)
print(history.history, file=open(f"NNBlock/experiments/{EXPERIMENT_NAME}/history.txt", "w"))