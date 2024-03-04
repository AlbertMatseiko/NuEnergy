import numpy as np
import h5py as h5
import time
import yaml
from pathlib import Path

try:
    from .dataset_from_h5 import DatasetInput, make_dataset
except:
    from dataset_from_h5 import DatasetInput, make_dataset

import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def load_ds_params(path_to_model_dir, yml_name="dataset_params"):
    path_to_yml = f"{path_to_model_dir}/{yml_name}.yaml"
    ds_dict = yaml.full_load(Path(path_to_yml).read_text())
    return DatasetInput(**ds_dict)

def load_log10e_normp(path_to_h5):
    with h5.File(path_to_h5, 'r') as hf:
        log10E_mean = hf["norm_params/log10Emu_mean"][:]
        log10Emu_std = hf["norm_params/log10Emu_std"][:]
    return log10E_mean, log10Emu_std

def make_preds(path_to_model_dir, model_regime="best_by_test", ds_regime="val", bath_size=1024):
    time0 = time.time()
    # load model
    model = tf.saved_model.load(f"{path_to_model_dir}/{model_regime}")

    # create dataset with logged parameters, get some info
    ds_inp = load_ds_params(path_to_model_dir)
    ds_inp.batch_size = bath_size
    ds, bs, total_num = make_dataset(ds_regime, ds_inp)
    assert bs == bath_size

    # make numpy generator from dataset
    gen = ds.as_numpy_iterator()

    # initialize preds, labels and weights
    preds = np.zeros((total_num // bs * bs, 2))
    labels = np.zeros((total_num // bs * bs, 2))
    weights = np.zeros((total_num // bs * bs, 1))

    # fill preds with model calls, labels with true values and also fill weights
    for j in range(total_num // bs):
        if j % 1000 == 0:
            print(f"Step {j} out of {total_num // bs}")
        data, y_true, w = gen.next()
        preds[j * bs:(j + 1) * bs] = model(data)
        labels[j * bs:(j + 1) * bs] = y_true
        weights[j * bs:(j + 1) * bs] = w
    time1 = time.time()

    # send predictions to h5 file
    with h5.File(f"{path_to_model_dir}/{model_regime}/Predictions_{ds_regime}.h5", 'w') as hfout:
        # dirs for predictions
        hfout.create_dataset(f"preds", data=preds)
        hfout.create_dataset(f"labels", data=labels)
        hfout.create_dataset(f"weights", data=weights)

        # dir for meta information of the process
        hfout.create_dataset(f"time_batchsize_wastotalnum",
                             data=np.array([time1 - time0, bs, total_num], dtype=np.float32))

    return preds, labels, weights

def load_preds(path_to_model_dir, model_regime="best_by_test", ds_regime="val",
               renorm: bool = True):
    # load predictions from model dir, that should be made in advance
    with h5.File(f"{path_to_model_dir}/{model_regime}/Predictions_{ds_regime}.h5", 'r') as hf:
        preds = hf["preds"][:]
        labels = hf["labels"][:]
        weights = hf["weights"][:]

    # Do you want to get log10E itself?
    if renorm:
        ds_inp = load_ds_params(path_to_model_dir)
        mean, std = load_log10e_normp(ds_inp.path_to_h5)
        preds[:, 0] = preds[:, 0] * std + mean
        preds[:, 1] = preds[:, 1] * std
        labels[:, 0] = labels[:, 0] * std + mean

    return preds, labels, weights