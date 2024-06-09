import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import tensorflow as tf
from NNBlock.launcher import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

EXPERIMENT_NAME = "08_06_BigResNet"
DESCRIPTION = """Big ResNet Encoder on new flat dataset with tails. WITH gauss noise. Normal batch size. Softplus actvation for sigma."""
CONTINUE_TRAINING = False

# starting training process with configs described in yml_configs directory
model, history = launch_exp(EXPERIMENT_NAME, 
                            DESCRIPTION=DESCRIPTION, 
                            CONTINUE_TRAINING=CONTINUE_TRAINING)
#print(history.history, file=open(f"NNBlock/experiments/{EXPERIMENT_NAME}/history.txt", "w"))