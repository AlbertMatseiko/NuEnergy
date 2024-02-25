import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from launcher import *

EXPERIMENT_NAME = "JustTest14"
MODEL_FUNC_NAME = "TwoTapesModel"
history = launch_exp(EXPERIMENT_NAME, MODEL_FUNC_NAME)
print(history.history)
