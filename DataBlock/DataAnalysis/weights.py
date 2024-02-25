import numpy as np
import h5py as h5


def get_dir_name(path_to_h5):
    return path_to_h5[: len(path_to_h5) - path_to_h5[::-1].index('/') - 1]


def get_step_name(path_to_h5):
    return path_to_h5[len(path_to_h5) - path_to_h5[::-1].index('/'):]


def calc_weights_distr(path_to_h5: str, regime='train'
                       , bins=50):

    with h5.File(f"{path_to_h5}", 'r') as hf:
        energy = hf[f'{regime}/log10Emu_norm'][:]
    dens, e_bins = np.histogram(energy, bins=bins, density=True)
    weights_distr = max(dens) / dens
    e_bins_av = (e_bins[:-1] + e_bins[1:]) / 2
    weights_hist = np.concatenate([weights_distr[:, np.newaxis], e_bins_av[:, np.newaxis]], axis=-1)

    path_to_h5_dir = get_dir_name(path_to_h5)
    np.save(f"{path_to_h5_dir}/weights_distr_{regime}.npy", weights_hist)
    return weights_hist
