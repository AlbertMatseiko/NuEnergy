import os
import sys

path_to_project = "/NuEnergy"
sys.path.append(path_to_project)

import numpy as np
import tensorflow as tf
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import clear_output
from scipy.stats import norm

import absolute_paths.PATHS_MuEnergy as paths
from customs.ds_making import make_dataset

try:
    # GPU on
    gpus = tf.config.list_physical_devices('GPU')
    print("The gpu' are:")
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
# mean of log10 energies in current train h5 file
MEAN, STD = 3.782758, 1.0855767


class Analysis:
    def __init__(self, exp_name, regime='val',
                 path_to_project=path_to_project,
                 path_to_h5=paths.path_to_h5,
                 logE_mean=MEAN, logE_std=STD,
                 version="best_by_test"):

        self.w = None
        self.sigma_true = None
        self.sigma_pred = None
        self.logE_pred = None
        self.logE_true = None
        self.energy_true = None
        self.preds = None

        self.exp_name = exp_name
        self.regime = regime
        self.version = version
        self.path_to_h5 = path_to_h5
        self.path_to_project = path_to_project

        self.path_to_exp = f"{self.path_to_project}/experiments/{self.exp_name}"
        self.path_to_model = f"{self.path_to_exp}/{version}"
        self.path_to_preds = f"{self.path_to_exp}/predictions/preds_{self.regime}_{version}.npy"

        self.logE_mean = logE_mean
        self.logE_std = logE_std

        self.dist_arr = np.load(paths.path_to_dist_arr)
        self.ds = None
        self.model = None

        # self.logE_pred = preds[:, 0] * STD + MEAN

    def load_model(self):
        self.model = tf.saved_model.load(self.path_to_model)
        clear_output()

    # making predictions and saving
    def make_preds(self, bs=1024, shape=(None, 6), apply_add_gauss=True, apply_mult_gauss=True, apply_Q_log=False):
        self.ds = make_dataset(self.path_to_h5, self.regime, batch_size=bs, shape=shape,
                               apply_add_gauss=apply_add_gauss, apply_mult_gauss=apply_mult_gauss,
                               apply_Q_log=apply_Q_log)
        if self.model is None:
            self.load_model()
        with h5.File(self.path_to_h5, 'r') as hf:
            L = hf[f'{self.regime}/ev_starts/data'][:].shape[0] - 1
        gen = self.ds.as_numpy_iterator()
        self.preds = np.zeros((L // bs * bs, 2))  # set zero by default
        for j in range(L // bs):
            print(f"Step {j} out of {L // bs}")
            data, labels = gen.next()
            self.preds[j * bs:(j + 1) * bs] = self.model(data)
            clear_output()
        # self.model.predict(self.ds, steps=L//bs, verbose=False)
        os.makedirs(f"{self.path_to_exp}/predictions", exist_ok=True)
        np.save(self.path_to_preds, self.preds)
        return 0

    # to load predictions from the path
    def load_preds(self):
        try:
            self.preds = np.load(self.path_to_preds)
            self.logE_pred = self.preds[:, 0] * self.logE_std + self.logE_mean

            # check if it was IsSigma regime
            if "IsSigmaTrue" in self.exp_name:  # maybe should change to "if preds[:,0]==preds[:,1]"
                self.sigma_pred = self.preds[:, 1] * self.logE_std
            else:
                self.preds[:, 1] = np.ones((self.preds.shape[0],))
                self.sigma_pred = np.copy(self.preds[:, 1])
            return 0
        except:
            print(f"There are no preds for {self.exp_name}. Program termination.")
            return 1

    # load true energy values from h5
    def load_E_true(self):
        with h5.File(self.path_to_h5, 'r') as hf:
            self.energy_true = hf[f'{self.regime}/muons_prty/individ/data'][:, -1]
            #self.energy_true = hf[f'{self.regime}/prime_prty/data'][:, 2]
        self.logE_true = np.log10(self.energy_true)
        return 0

    # load true energy values from h5 and calculate true sigmas from preds
    def load_true(self):
        self.load_E_true()
        self.load_preds()
        self.sigma_true = np.abs(self.logE_true[:self.logE_pred.shape[0]] - self.logE_pred)
        return 0

    # get weight for the energy true E according to distribution
    def w_func(self, y_E):
        y_E_bin = np.reshape(self.dist_arr[0:1], (1, np.shape(self.dist_arr)[1]))
        y_E = np.expand_dims(y_E, axis=-1)
        j = np.argmin(np.abs(y_E_bin - y_E), axis=-1)
        return self.dist_arr[1][j]

    # load weights for events as it were used in train
    def load_weights(self):
        # that's right, apply weiths to NORMED LOG10 Energies
        self.w = self.w_func((self.logE_true - self.logE_mean) / self.logE_std)
        return 0

    # plot hists of logE for true and preds
    def plot_logE_hist(self, do_weights=False):
        if self.logE_pred is None:
            self.load_preds()
        if self.logE_true is None:
            self.load_E_true()

        # loading weights if necessary
        if do_weights:
            self.load_weights()
            weights = self.w[:self.logE_pred.shape[0]]
        else:
            weights = None

        # plotting

        x_TrueE = (self.logE_true[:self.logE_pred.shape[0]], r'True $\log_{10}E$')
        x_PredE = (self.logE_pred, r'Predicted $\log_{10}E$')
        data_list = [x_TrueE, x_PredE]
        labels = [r"True $\log_{10}E$ distibution", r"Predicted $\log_{10}E$ distibution"]
        colors = ["green", "coral"]

        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        hist_kwargs = dict(weights=weights, bins=100, density=True)
        tick_size = 20
        label_size = 30
        title = f"HistsLog10E_{self.regime}_Weights{do_weights}_{self.version}"
        fig.suptitle(title, fontsize=30)

        def plot_one_hist(a, data: tuple, color="green", hist_title=""):
            x_data = data[0]
            x_label = data[1]
            l = a.hist(x_data, color=color, **hist_kwargs)
            a.set_xlim(0., x_data.max() + 0.5)
            a.set_xlabel(x_label, fontsize=label_size)
            a.set_ylabel('Density', fontsize=label_size)
            a.xaxis.set_tick_params(labelsize=tick_size)
            a.yaxis.set_tick_params(labelsize=tick_size)
            a.grid(linestyle=":")
            a.set_title(hist_title, fontsize=label_size)

        for i in range(2):
            plot_one_hist(ax[i], data=data_list[i], color=colors[i], hist_title=labels[i])
        os.makedirs(f"{self.path_to_exp}/figures", exist_ok=True)
        plt.savefig(f"{self.path_to_exp}/figures/{title}.png")
        plt.close()

        return fig

    def plot_hists2d(self, size=None, do_weights=False):
        # loading preds and true if necessary
        if self.preds is None:
            self.load_preds()
        if self.sigma_true is None:
            self.load_true()

        # loading weights if necessary
        if do_weights:
            self.load_weights()
            weights = self.w[:self.logE_pred.shape[0]]
            # w_pred = self.w_pred
        else:
            weights = None
            # w_pred = None

        # making slices on data to have both atmospheric and astrophysics neutrino
        if size is not None:
            idxs = [i for i in range(size // 2)] + [len(self.logE_pred) - i - 1 for i in range(size // 2)]
        else:
            idxs = [i for i in range(len(self.logE_pred))]
        x_E = (self.logE_true[idxs], r"True $\log_{10}E$")
        y_E = (self.logE_pred[idxs], r"Predicted $\log_{10}E$")
        x_s = (self.sigma_true[idxs], r"True error")
        y_s = (self.sigma_pred[idxs], r"Predicted $\sigma$")
        data_list = [[(x_E, y_E), (x_s, y_s)],
                     [(y_E, y_s), (x_E, y_s)],
                     [(y_E, x_s), (x_E, x_s)]]
        labels = [[r'Hist2D $\log_{10}E$', r'Hist2D $\sigma$'],
                  [r'$\sigma_{pred}$ vs $\log_{10}E_{pred}$', r'$\sigma_{pred}$ vs $\log_{10}E_{true}$'],
                  [r'$\sigma_{true}$ vs $\log_{10}E_{pred}$', r'$\sigma_{true}$ vs $\log_{10}E_{true}$']]

        # plotting
        fig, ax = plt.subplots(3, 2, figsize=(30, 45))
        tick_size = 20
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        label_size = 30
        title = f"Hist2D_{self.regime}_Weight{do_weights}_{self.version}"
        fig.suptitle(title, fontsize=label_size)

        hist_kwargs = dict(weights=weights, bins=50, density=True, cmap=plt.cm.jet, norm=colors.LogNorm())

        def plot_one_hist(a, data: tuple, hist_title="", add_line=False):
            x_data, y_data = data[0][0], data[1][0]
            x_label, y_label = data[0][1], data[1][1]
            l = a.hist2d(x_data, y_data, **hist_kwargs)
            if add_line:
                x = np.linspace(0, x_data.max() + 0.5, 10)
                l0 = a.plot(x, x, '-', c='black')
            a.set_xlim(0., x_data.max() + 0.5)
            a.set_ylim(0., y_data.max() + 0.5)
            a.set_xlabel(x_label, fontsize=label_size)
            a.set_ylabel(y_label, fontsize=label_size)
            a.xaxis.set_tick_params(labelsize=tick_size)
            a.yaxis.set_tick_params(labelsize=tick_size)
            a.grid(linestyle=":")
            a.set_title(hist_title, fontsize=label_size)

        for i in range(3):
            for j in range(2):
                ax_cur = ax[i, j]
                plot_one_hist(ax_cur, data_list[i][j], hist_title=labels[i][j], add_line=(i == 0))

        os.makedirs(f"{self.path_to_exp}/figures", exist_ok=True)
        plt.savefig(f"{self.path_to_exp}/figures/{title}.png")
        plt.close()

        return fig

    def plot_x(self):# Plot between -10 and 10 with .001 steps.
        x_axis = np.arange(-10, 10, 0.001)
        y_true = self.logE_true[:self.preds.shape[0]]  # (self.logE_true[:self.preds.shape[0]]-self.logE_mean)/(self.logE_std) #
        y_pred = self.logE_pred  # self.preds[:,0]#
        sigma = self.sigma_pred  # self.preds[:,1]#
        values = (y_true - y_pred) / sigma
        values = values[sigma > 0.01]
        # values = values[(y_true>0)*(y_true<10)]
        # self.load_weights()
        fig, ax = plt.subplots(figsize=(10, 10))  # = plt.figure()
        plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="Нормальное распределение, N(0,1)")
        h = plt.hist(values, bins=1000, density=True,
                     label="Распределение отклонений предсказаний, \nделённых на предсказанную погрешность")  # , weights=self.w[:len(values)])#, log=True)#,  range=(-5.,5.))
        plt.grid(which='both', linestyle=':')
        plt.xlabel(r"x=$\frac{(\log_{10}E_{true}-\log_{10}E*)}{\sigma*}$", fontsize=20)
        plt.ylabel(r"Density", fontsize=15)
        # plt.ylim(1e-6,0.5)
        plt.xlim(-10, 10)
        ax.minorticks_on()
        plt.legend()
        title = f"x {self.regime} distibution {self.version}"
        # plt.title(title)
        plt.savefig(f"{self.path_to_exp}/figures/{title}.png")
        plt.close()
        return fig
