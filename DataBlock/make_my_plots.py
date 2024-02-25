import numpy as np
import h5py as h5
from pathlib import Path

from DataAnalysis.plot_data import plot_data_hists


def make_plots(h5_dir, h5_step, list_of_keys=None, Q_max=0.2):
    if list_of_keys is None:
        list_of_keys = ['train', 'test', 'val']
    path_to_save = Path(__file__).parent.absolute() / 'data' / h5_dir
    path_to_h5 = path_to_save / h5_step
    with h5.File(path_to_h5, 'r') as hf:
        for regime in list_of_keys:
            data = hf[f"{regime}/data"][:]
            lens = np.diff(hf[f"{regime}/ev_starts"][:])
            plot_data_hists(data[:], bins=200, len_ev=lens, Q_max=Q_max,
                            path_to_save=path_to_save, title=f'{h5_dir}_{regime}_data_distr')


if __name__ == '__main__':
    h5_dir = "baikal_multi_1223_flat_doubles_H8_Q2_S2"
    #h5_step, list_of_keys, Q_max = "step3_all_normed.h5", ['train', 'test', 'val'], 0.2
    h5_step, list_of_keys, Q_max = "step1.h5", ['nuatm', 'nu2'], 30
    make_plots(h5_dir, h5_step, list_of_keys, Q_max)
