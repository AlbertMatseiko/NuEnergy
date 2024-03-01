import sys
sys.path.append("/home/albert/Baikal/NuEnergy")

import numpy as np
import tensorflow as tf
import h5py as h5

def make_preds(self, bs=1024, shape=(None, 6), apply_mult_gauss=True):