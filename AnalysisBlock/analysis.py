import h5py as h5
import os
import numpy as np
import tensorflow as tf
from scipy.stats import beta
import plotly.graph_objects as go
from NuEnergy.NNBlock.dataset_from_h5 import make_dataset


class Analysis:
    def __init__(self, model, model_name, path_to_h5=None, regime='val', tr_start=0., tr_end=1., N_points=10000,
                 proba_nn=None, path_to_preds=None, path_to_positives: str = './preds_analysis/positives'):
        self.mu_nu_ratio = None
        self.path = None
        self.model = model
        self.mn = model_name #self.model._name
        self.path_to_h5 = path_to_h5
        self.regime = regime
        self.proba_nn = proba_nn
        self.path_to_preds = path_to_preds
        self.preds_mu = None
        self.preds_nuatm = None
        self.preds_nue2 = None
        self.preds_nu = None
        self.labels = None
        self.path_to_positives = path_to_positives
        self.tr_start = tr_start
        self.tr_end = tr_end
        self.N_points = N_points
        self.tr_arr = np.linspace(self.tr_start, self.tr_end, self.N_points)
        self.MuPos, self.NuPos = None, None
        self.E, self.S = None, None
        self.nu_in_flux = None
        self.preds_mu_flux = None
        self.preds_mu_test = None
        self.MuPos_flux = None
        self.MuPos_test = None
        self.preds_nu_flux = None
        self.preds_nu_test = None
        self.NuPos_flux = None
        self.NuPos_test = None
        self.S_test = None
        self.E_test = None
        self.NuFromNN = None
        self.sigma_S = None
        self.sigma_E = None
        self.S1 = None
        self.S2 = None
        self.S3 = None
        self.sigma_N = None

        if path_to_preds is not None:
            assert path_to_preds.endswith(self.regime + '.npy')

    def make_preds(self, shape=(None, 6), bs=1024):
        assert self.path_to_h5 is not None
        assert self.model is not None
        with h5.File(self.path_to_h5, mode='r') as hf:
            L = len(hf[self.regime + '/ev_ids_corr/data'])
        dataset = make_dataset(self.path_to_h5, regime=self.regime, batch_size=bs, shape=shape)
        self.proba_nn = self.model.predict(dataset, steps=L // bs)

        try:
            os.makedirs('preds_analysis/predictions')
            print('directory for preds is created')
        except:
            print('directory for preds already exists')
        self.path_to_preds = './preds_analysis/predictions/preds_' + self.mn + '_' + self.regime
        np.save(self.path_to_preds, self.proba_nn)

        return self.path_to_preds

    def load_preds_and_labels(self):
        #if self.path_to_preds is None:
        self.path_to_preds = './preds_analysis/predictions/preds_' + self.mn + '_' + self.regime + '.npy'
        self.proba_nn = np.load(self.path_to_preds)
        with h5.File(self.path_to_h5, 'r') as hf:
            L = self.proba_nn.shape[0]
            self.labels = np.zeros((L, 1))
            ids = hf[self.regime + '/ev_ids_corr/data'][0:L]
            ids_mu = np.where(np.char.startswith(ids, b'mu'))[0]
            ids_nuatm = np.where(np.char.startswith(ids, b'nuatm'))[0]
            ids_nue2 = np.where(np.char.startswith(ids, b'nue2'))[0]
            self.labels[ids_mu] = 0.
            self.labels[ids_nuatm] = 1.
            self.labels[ids_nue2] = 2.
        return self.proba_nn[:, 1], self.labels[:, 0]

    def separate_preds(self):
        idxs_mu = np.where(self.labels == 0)[0]
        idxs_nuatm = np.where(self.labels == 1)[0]
        idxs_nue2 = np.where(self.labels == 2)[0]
        self.preds_mu = self.proba_nn[idxs_mu]
        self.preds_nuatm = self.proba_nn[idxs_nuatm]
        self.preds_nue2 = self.proba_nn[idxs_nue2]
        self.preds_nu = np.concatenate([self.preds_nuatm, self.preds_nue2], axis=0)
        return self.preds_mu, self.preds_nu

    def _get_positive(self, tr_arr, predictions, batch_size=2048 * 32):
        if predictions.shape[-1] == 2:
            preds = predictions[:, 1].reshape(predictions.shape[0], 1)
        elif predictions.shape[-1] != 1:
            preds = predictions.reshape(predictions.shape[0], 1)
        else:
            preds = predictions
        pos = tf.zeros(tr_arr.shape[0], dtype=tf.int32)
        bs = batch_size
        last = 0
        if preds.shape[0] >= bs:
            L = preds.shape[0] // bs
            print(L)
            for i in range(L):
                x = tf.reduce_sum(tf.where(preds[i * bs:(i + 1) * bs, 0:1] >= tr_arr, 1, 0), axis=0)
                pos = pos + x
            last = L * bs
        x = tf.reduce_sum(tf.where(preds[last:, 0:1] >= tr_arr, 1, 0), axis=0)
        pos = pos + x
        return pos.numpy()

    def get_pos_rates(self, batch_size=2048 * 32,
                                     update_positives: bool = False,
                                     postfix=''):
        self.path = f"{self.path_to_positives}/{self.mn}_{self.regime}_{self.tr_start}_{self.tr_end}_{self.N_points}"
        try:
            assert not update_positives
            self.MuPos = np.load(self.path + '_MuPos.npy')
            self.NuPos = np.load(self.path + '_NuPos.npy')
        except:
            self.MuPos = self._get_positive(self.tr_arr, self.preds_mu, batch_size=batch_size)
            print('Got Mu!')
            np.save(self.path + '_MuPos' + postfix + '.npy', self.MuPos)
            self.NuPos = self._get_positive(self.tr_arr, self.preds_nu, batch_size=batch_size)
            np.save(self.path + '_NuPos' + postfix + '.npy', self.NuPos)
            print('Got Nu!')
        self.E = self.NuPos / self.preds_nu.shape[0]
        self.S = self.MuPos / self.preds_mu.shape[0]
        return self.E, self.S

    def plot_SE(self, scale_y: str = 'linear', scale_x: str = 'linear'):
        # assert self.tr_start == 0. and self.tr_end == 1.
        if self.E is None or self.S is None:
            self.E, self.S = self.get_pos_rates()

        fig = go.Figure(go.Scatter(x=self.tr_arr, y=self.S, name=f"Suppression", mode="lines",
                                   line=go.scatter.Line(color='coral')))
        fig.add_scatter(x=self.tr_arr, y=self.E, mode="lines", name=f"Exposition",
                        line=go.scatter.Line(color='lightskyblue'))
        fig.update_layout(width=600, height=600, title="E and S vs classification threshold", title_x=0.5,
                          xaxis_title="Threshold", yaxis_title="E and S values")
        if scale_y == 'log':
            range_y = [-8, 0]
            range_x = [0.9, 1.001]
        else:
            range_y = [0., 1.001]
            range_x = [0., 1.001]
        fig.update_yaxes(type=scale_y, range=range_y)
        fig.update_xaxes(type=scale_x, range=range_x)
        fig.show()
        return fig

    def get_pos_rates_flux(self, batch_size=2048 * 32,
                                     update_positives: bool = False,
                                     postfix=''):
        self.path = f"{self.path_to_positives}/{self.mn}_{self.regime}_{self.tr_start}_{self.tr_end}_{self.N_points}"
        try:
            assert not update_positives
            self.MuPos_test = np.load(self.path + '_MuPos_test'+postfix+'.npy')
            self.MuPos_flux = np.load(self.path + '_MuPos_flux'+postfix+'.npy')
            self.NuPos_test = np.load(self.path + '_NuPos_test'+postfix+'.npy')
            self.NuPos_flux = np.load(self.path + '_NuPos_flux'+postfix+'.npy')
        except:
            self.MuPos_test = self._get_positive(self.tr_arr, self.preds_mu_test, batch_size=batch_size)
            self.MuPos_flux = self._get_positive(self.tr_arr, self.preds_mu_flux, batch_size=batch_size)
            print('Got Mu!')
            np.save(self.path + '_MuPos_test' + postfix + '.npy', self.MuPos_test)
            np.save(self.path + '_MuPos_flux' + postfix + '.npy', self.MuPos_flux)
            self.NuPos_test = self._get_positive(self.tr_arr, self.preds_nu_test, batch_size=batch_size)
            self.NuPos_flux = self._get_positive(self.tr_arr, self.preds_nu_flux, batch_size=batch_size)
            np.save(self.path + '_NuPos_test' + postfix + '.npy', self.NuPos_test)
            np.save(self.path + '_NuPos_flux' + postfix + '.npy', self.NuPos_flux)
            print('Got Nu!')
        self.E_test = self.NuPos_test / self.preds_nu_test.shape[0]
        self.S_test = self.MuPos_test / self.preds_mu_test.shape[0]
        return self.E_test, self.S_test

    def get_NuFromNN(self, nu_in_flux=30, mu_nu_ratio=1e5, start_mu=0, start_nu=0, alpha=1. - 0.68,
                     update_positives: bool = False):
        assert self.preds_mu is not None and self.preds_nu is not None
        self.nu_in_flux = nu_in_flux
        self.mu_nu_ratio = mu_nu_ratio
        if self.preds_mu.shape[0] / nu_in_flux < mu_nu_ratio:
            print(f"mu_nu_ration is too huge! Max is {self.preds_mu.shape[0] / nu_in_flux}. Try again.")
            return self.preds_mu.shape[0] / nu_in_flux
        else:
            len_mu = int(mu_nu_ratio * nu_in_flux)
            self.preds_mu_flux = self.preds_mu[start_mu:start_mu + len_mu]
            self.preds_mu_test = np.concatenate([self.preds_mu[start_mu + len_mu:],
                                                 self.preds_mu[0:start_mu]], axis=0)

            len_nu = nu_in_flux
            self.preds_nu_flux = self.preds_nu[start_nu:start_nu + len_nu]
            self.preds_nu_test = np.concatenate([self.preds_nu[start_nu + len_nu:],
                                                 self.preds_nu[0:start_nu]], axis=0)

            self.E_test, self.S_test = self.get_pos_rates_flux(postfix=f"_{nu_in_flux}_{mu_nu_ratio}_{start_mu}_{start_nu}",
                                                               update_positives=update_positives)

            ### Формула потока
            n_0 = self.preds_mu_flux.shape[0] + self.preds_nu_flux.shape[0]
            n_xi = self.MuPos_flux + self.NuPos_flux
            self.NuFromNN = (n_xi - self.S_test * n_0) / (self.E_test - self.S_test)
            ### Оценка ошибки
            self.sigma_S, self.sigma_E = [], []
            # Считаем погрешность S
            n = self.preds_mu_test.shape[0]  # MuPos_test[0]
            for k in self.MuPos_test:
                low, up = beta.ppf([alpha / 2, 1 - alpha / 2],
                                   [k, k + 1],
                                   [n - k + 1, n - k])
                low = np.nan_to_num(low)
                self.sigma_S.append((up - low) / 2)  # (max(k/n - low, up - k/n))

            # Считаем погрешность E
            n = self.preds_nu_test.shape[0]  # NuPos_test[0]
            for k in self.NuPos_test:
                low, up = beta.ppf([alpha / 2, 1 - alpha / 2],
                                   [k, k + 1],
                                   [n - k + 1, n - k])
                up = np.nan_to_num(up, nan=1.0)
                self.sigma_E.append((up - low) / 2)

            # Считаем погрешность формулы для N
            self.sigma_S, self.sigma_E = np.array(self.sigma_S), np.array(self.sigma_E)
            self.S1 = (n_xi - self.S_test * n_0) ** 2 / (self.E_test - self.S_test) ** 4 * self.sigma_E ** 2
            self.S2 = (n_xi - self.E_test * n_0) ** 2 / (self.E_test - self.S_test) ** 4 * self.sigma_S ** 2
            self.S3 = 0  # ((1-2*self.S_test)*(n_xi+n_0*self.S_test**2)/(self.E_test-self.S_test)**2
            self.sigma_N = np.sqrt(self.S1 + self.S2 + self.S3)

            return self.NuFromNN, self.sigma_N

    def plot_error_and_flux(self, cut_low=0.5, cut_up=0.99999):
        if self.NuFromNN is None:
            self.NuFromNN, self.sigma_N = self.get_NuFromNN()
        start = int(self.N_points * cut_low)
        end = int(self.N_points * cut_up)
        x = self.tr_arr[start:end]
        y = self.NuFromNN[start:end]
        y_true = np.array(([self.nu_in_flux] * self.N_points)[start:end])
        y_up = y_true + self.sigma_N[start:end]
        y_low = y_true - self.sigma_N[start:end]
        fig = go.Figure(go.Scatter(x=x, y=y, name=f"Number of nu by formula", mode="lines",
                                   line=go.scatter.Line(color='lightskyblue')))
        fig.add_scatter(x=x, y=y_up, mode="lines", name=f"Error limits",
                        line=go.scatter.Line(color='red', dash='dot'))
        fig.add_scatter(x=x, y=y_low, mode="lines", name=f"Error limits",
                        line=go.scatter.Line(color='red', dash='dot'))
        fig.add_scatter(x=x, y=y_true, mode="lines", name=f"True Nu number",
                        line=go.scatter.Line(color='green', dash='solid'))
        fig.update_layout(width=800, height=600, title="The evaluation of neutrino flux", title_x=0.5,
                          xaxis_title="Threshold", yaxis_title="Nu number")
        fig.show()
        return fig
