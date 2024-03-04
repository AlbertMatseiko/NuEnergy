import numpy as np
import h5py as h5
import tensorflow as tf
from dataclasses import dataclass, asdict
import yaml
from pathlib import Path


### Defaults
# gauss noise
# 0.1 ~ 1 p.e., 150 ns, 4, 4, 15 m
# apply_add_gauss = True
# # normal
# g_add_stds = [0.03, 0.005, 0.005, 0.005, 0.00003]
# # bigger
# # stds_gauss = [0.03, 0.005, 0.02, 0.02, 0.0007]
# apply_mult_gauss = True
# q_noise_fraction = 0.1
#
# # limiting Q vals
# set_up_Q_lim = True
# up_Q_lim = 100

def get_dir_name(path_to_h5):
    return path_to_h5[: len(path_to_h5) - path_to_h5[::-1].index('/') - 1]


def get_weights_tf(nlogE, weights_hist):
    """
    Function to calculate weights for given batch of energies in dataset
    with response to weights histogram
    using tensorflow
    """
    weights_hist = tf.cast(weights_hist, tf.float32)
    nlogE_bin = weights_hist[:, 1:2]  # these are NORMED LOG10 Energies
    weights_arr = weights_hist[:, 0:1]
    nlogE = tf.cast(nlogE, tf.float32)
    j = tf.argmin(tf.abs(tf.transpose(nlogE_bin) - nlogE), axis=-1)
    return tf.gather(weights_arr, indices=j)


# mult noise
class gauss_mult_noise:
    def __init__(self, Q_mean_noise, n_fraction):
        self.Q_mean_noise = Q_mean_noise
        self.n_fraction = n_fraction

    def make_noise(self, Qs):
        noises = np.random.normal(scale=self.n_fraction, size=Qs.shape)
        Qs = Qs + (Qs + self.Q_mean_noise) * noises
        return Qs


class generator:
    def __init__(self, path_to_h5, regime, batch_size,
                 set_up_Q_lim, up_Q_lim,
                 apply_add_gauss, g_add_stds, apply_mult_gauss, q_noise_fraction, apply_Q_log=False, start=0,
                 use_weights=True):

        self.path_to_h5 = path_to_h5
        self.regime = regime
        self.batch_size = batch_size
        self.start = start

        self.hf = h5.File(self.path_to_h5, 'r')
        hf = self.hf

        self.ev_starts = hf[regime + '/ev_starts']
        self.num = len(self.ev_starts[1:] - self.ev_starts[0:-1])
        self.batch_num = self.num // self.batch_size
        self.gen_num = self.batch_num * self.batch_size

        masking_values = np.array([0., 1e5, 1e5, 1e5, 1e5])  # нулеваой заряд, далёкое время и координаты
        self.norm_zeros = (masking_values - hf['norm_params/mean'][:]) / hf['norm_params/std'][:]

        # For noise
        self.set_up_Q_lim = set_up_Q_lim
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.g_add_stds = g_add_stds
        self.apply_Q_log = apply_Q_log

        self.Q_mean = hf['norm_params/mean'][0]
        self.Q_std = hf['norm_params/std'][0]
        self.Q_log_mean = np.log(hf['train/data'][:, 0] * self.Q_std + self.Q_mean).mean()
        self.Q_log_std = np.log(hf['train/data'][:, 0] * self.Q_std + self.Q_mean).std()
        self.Q_eps = 1
        if set_up_Q_lim:
            self.Q_up_lim_norm = (up_Q_lim - self.Q_mean) / self.Q_std
        if apply_mult_gauss:
            self.mult_gauss = gauss_mult_noise(self.Q_mean / self.Q_std, q_noise_fraction)
        self.batch_num = self.num // self.batch_size

        # load weights for energy distribution
        self.use_weights = use_weights
        path_to_h5_dir = get_dir_name(path_to_h5)
        self.dist_of_weights = tf.cast(np.load(f"{path_to_h5_dir}/weights_distr_train.npy"), tf.float32)

    # addative noise
    def add_gauss(self, data, g_add_stds, ev_starts):
        g_add_stds = np.broadcast_to(g_add_stds, data.shape)
        noise = np.random.normal(scale=g_add_stds, size=data.shape)
        data[:, 1:] += noise[:, 1:]
        # not allow non physical noise augmentaion: Q > 0 p.e.
        data[:, 0] = np.where((data[:, 0] + noise[:, 0]) * self.Q_std + self.Q_mean > 0.,
                              data[:, 0] + noise[:, 0],
                              data[:, 0])
        ev_idxs_local = ev_starts - ev_starts[0]
        sort_idxs = np.concatenate(
            [np.argsort(data[ev_idxs_local[i]:ev_idxs_local[i + 1], 1], axis=0) + ev_idxs_local[i] for i in
             range(len(ev_starts) - 1)]
        )
        data = data[sort_idxs]
        return data

    def step(self, start, stop, ev_starts):
        hf = self.hf
        data_start = ev_starts[0]
        data_stop = ev_starts[-1]
        data = hf[self.regime + '/data'][data_start: data_stop]
        labels = np.zeros((self.batch_size, 2))
        labels[:, 0:1] = hf[f'{self.regime}/log10Emu_norm'][start:stop]
        if self.set_up_Q_lim:
            data[:, 0:1] = np.where(data[:, 0:1] > self.Q_up_lim_norm, self.Q_up_lim_norm, data[:, 0:1])

        # apply noise
        if self.apply_add_gauss:
            data = self.add_gauss(data, self.g_add_stds, ev_starts)
        if self.apply_mult_gauss:
            data[:, 0] = self.mult_gauss.make_noise(data[:, 0])
        # apply log scale to Q
        if self.apply_Q_log:
            Q_log = np.log(data[:, 0] * self.Q_std + self.Q_mean)
            data[:, 0] = (Q_log - self.Q_log_mean) / self.Q_log_std
            self.norm_zeros[0] = (-1e6 - self.Q_log_mean) / self.Q_log_std

        check = 1
        while check:
            try:
                data = tf.RaggedTensor.from_row_starts(values=data, row_starts=ev_starts[0:-1] - ev_starts[0])
                check = 0
            except:
                check = 1
        data = data.to_tensor(default_value=self.norm_zeros)
        mask = tf.where(tf.not_equal(data[:, :, 1:2], self.norm_zeros[1:2]), 1., 0.)
        data = tf.concat([data, mask], axis=-1)

        if self.use_weights:
            weights = get_weights_tf(labels[:, 0:1], weights_hist=self.dist_of_weights)
            return data, labels, weights
        else:
            return data, labels, tf.ones((stop - start, 1), tf.float32)

    def __call__(self):
        start = self.start
        stop = self.start + self.batch_size
        for i in range(self.batch_num):
            ev_starts = self.hf[self.regime + '/ev_starts'][start:stop + 1]
            out_data = self.step(start, stop, ev_starts)
            #print(out_data[0].shape, file=open('./shapes.txt', "a"))
            yield out_data
            start += self.batch_size
            stop += self.batch_size


@dataclass
class DatasetInput:
    path_to_h5: str = ""
    batch_size: int = 32
    shape: tuple[int, int] = (None, 6)
    start: int = 0
    # limiting Q vals
    set_up_Q_lim: bool = True
    up_Q_lim: float = 100.
    apply_Q_log: bool = False
    # gauss noise
    # 0.1 ~ 1 p.e., 150 ns, 4, 4, 15 m
    apply_add_gauss: bool = False
    g_add_stds: list[float] = None
    apply_mult_gauss: bool = False
    q_noise_fraction: float = 0.1
    use_weights: bool = True


def make_dataset(regime, ds_input: DatasetInput = DatasetInput()):
    if ds_input.g_add_stds is None:
        # normal
        ds_input.g_add_stds = [0.03, 0.005, 0.005, 0.005, 0.00003]
        # bigger
        # ds_input.g_add_stds = [0.03, 0.005, 0.02, 0.02, 0.0007]
    bs = ds_input.batch_size
    gen_kwargs = {k: v for k, v in asdict(ds_input).items() if k not in ["shape"]}
    gen = generator(regime=regime, **gen_kwargs)
    dataset = tf.data.Dataset.from_generator(gen,
                                             output_signature=(tf.TensorSpec(shape=(bs,
                                                                                    ds_input.shape[0],
                                                                                    ds_input.shape[1]
                                                                                    )
                                                                             ),
                                                               tf.TensorSpec(shape=(bs, 2)),
                                                               tf.TensorSpec(shape=(bs, 1)))
                                             )
    if regime == 'train':
        dataset = dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # output is (<the dataset>, batch_size: int, total_num_of_events: int)
    return dataset, bs, gen.num
