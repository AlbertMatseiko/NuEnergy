import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import h5py as h5


def get_weights(nlogE, weights_hist):
    """
    Function to calculate weights for given batch of energies in dataset
    with response to weights histogram
    """
    if len(nlogE.shape)==1:
        nlogE = nlogE[:,np.newaxis]
    nlogE_bin = weights_hist[:, 1:2]  # these are NORMED LOG10 Energies
    weights_arr = weights_hist[:, 0:1]
    j = np.argmin(np.abs(nlogE_bin.T - nlogE), axis=-1)
    return weights_arr[j]

def plot_datasets_energy(path_to_h5, key='log10Emu_norm'):
    for r in ['train', 'test', 'val']:
        with h5.File(path_to_h5, 'r') as hf:
            Energy = hf[f"{r}/{key}"][:]
        path_to_dir = str(Path(path_to_h5).parent)
        weights_hist = np.load(f'{path_to_dir}/weights_distr_train.npy')
        w = get_weights(Energy, weights_hist)
        plt.hist(Energy, bins=50, color='green', density=True, label='Energy spectrum')
        plt.grid()
        plt.xlabel(f"log10E_norm, GeV")
        plt.ylabel(f"Density")
        plt.legend()
        plt.savefig(f'{path_to_dir}/{r}_{key}_Spec.png')
        plt.close()
        
        plt.hist(Energy, weights=w, bins=50, color='green', density=True, label='Weighted spectrum')
        plt.grid()
        plt.xlabel(f"log10E_norm, GeV")
        plt.ylabel(f"Density")
        plt.legend()
        plt.savefig(f'{path_to_dir}/{r}_weighted_{key}_Spec.png')
        plt.close()
    return None
    
# Функция для отрисовки
def plot_data_hists(data, len_ev=None, path_to_save=".",
                    title='Params distribution',
                    bins=100,
                    density=True,
                    Q_max=0.2,
                    show=False):
    fig, ax = plt.subplots(3, 2, figsize=(18, 18))

    fig.suptitle(title, fontsize=30)
    labels = ["Q", "t", "x", "y", "z", "Events' lengths"]

    ax0 = ax[0, 0]
    l0 = ax0.hist(data[np.abs(data[:, 0]) < Q_max, 0], bins=bins, density=density, color='blue')
    ax0.set_xlabel('Q', fontsize=20)
    ax0.set_ylabel('Density', fontsize=20)
    ax0.grid(":")

    ax1 = ax[0, 1]
    l1 = ax1.hist(data[:, 1], bins=bins, density=density, color='grey')
    ax1.set_xlabel('t', fontsize=20)
    ax1.set_ylabel('Density', fontsize=20)
    ax1.grid(":")

    ax2 = ax[1, 0]
    l2 = ax2.hist(data[:, 2], bins=bins, density=density, color='coral')
    ax2.set_xlabel('x', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.grid(":")

    ax3 = ax[1, 1]
    l3 = ax3.hist(data[:, 3], bins=bins, density=density, color='red')
    ax3.set_xlabel('y', fontsize=20)
    ax3.set_ylabel('Density', fontsize=20)
    ax3.grid(":")

    ax4 = ax[2, 0]
    l4 = ax4.hist(data[:, 4], bins=bins, density=density, color='green')
    ax4.set_xlabel('z', fontsize=20)
    ax4.set_ylabel('Density', fontsize=20)
    ax4.grid(":")

    if type(len_ev) is np.ndarray:
        ax5 = ax[2, 1]
        l5 = ax5.hist(np.log10(len_ev), bins=bins, density=density)
        ax5.set_xlabel('Lengths of events, log10', fontsize=20)
        ax5.set_ylabel('Density', fontsize=20)
        ax5.grid(":")
    else:
        l5 = None

    fig.legend([l0, l1, l2, l3, l4, l5], labels=labels, fontsize=20)
    if show:
        plt.show()

    # making dir for fig if necessary
    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(f'{path_to_save}/{title}.png')
    plt.close()
    return fig
