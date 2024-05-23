import numpy as np
import h5py as h5
import tensorflow as tf
from dataclasses import dataclass, asdict


class DsGeneratorNE:
    
    @staticmethod
    def get_dir_name(path_to_h5):
        return path_to_h5[: len(path_to_h5) - path_to_h5[::-1].index('/') - 1]
    
    @staticmethod
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
    
    # additive noise
    def _add_gauss(self, data, ev_starts):
        g_add_stds = np.broadcast_to(self.g_add_stds, data.shape)
        noise = np.random.normal(scale=g_add_stds, size=data.shape)
        data[:, 1:] += noise[:, 1:]
        # not allow non physical noise augmentaion: Q > 0 p.e.
        data[:, 0] = np.where((data[:, 0] + noise[:, 0]) * self.stds[0] + self.means[0] > 0.,
                              data[:, 0] + noise[:, 0],
                              data[:, 0])
        ev_idxs_local = ev_starts - ev_starts[0]
        sort_idxs = np.concatenate(
            [np.argsort(data[ev_idxs_local[i]:ev_idxs_local[i + 1], 1], axis=0) + ev_idxs_local[i] for i in
             range(len(ev_starts) - 1)]
        )
        data = data[sort_idxs]
        return data
    
    def _mult_gauss(self, Q_norm: np.ndarray[float]) -> np.ndarray[float]:
        """
        We want:
        Q' = Q * (1+q_fraction)
        In terms of normed charges:
        (Q'_norm * std + mean) = (Q_norm * std + mean) * (1+q_fraction)
        The function calculates Q'_norm from the formula and returns it!
        """
        noises = np.random.normal(loc=0, scale=self.q_fraction, size=Q_norm.shape)
        Q_norm_noised = Q_norm*(1+noises) + self.means[0]/self.stds[0] * noises
        return Q_norm_noised
    
    def __init__(self, path_to_h5, regime, batch_size,
                 set_up_Q_lim, up_Q_lim,
                 apply_add_gauss, g_add_stds, apply_mult_gauss, q_noise_fraction, start=0,
                 use_weights=False):

        # generation parameters
        self.path_to_h5 = path_to_h5
        self.regime = regime
        self.batch_size = batch_size
        self.start = start

        # file opening
        self.hf = h5.File(self.path_to_h5, 'r')
        hf = self.hf

        # meta info
        self.num = len(hf[f'{self.regime}/ev_starts'][1:] - hf[f'{self.regime}/ev_starts'][0:-1])
        self.batch_num = self.num // self.batch_size
        self.gen_num = self.batch_num * self.batch_size

        # load norming params
        self.means = self.hf[f'norm_params/mean'][:]
        self.stds = self.hf[f'norm_params/std'][:]
        self.lgEmean, self.lgEstd = self.hf[f'norm_params/log10Emu_mean'][:], self.hf[f'norm_params/log10Emu_std'][:]
        
        # перенормировка: нулеваой заряд, далёкое время и координаты
        masking_values = np.array([0., 1e5, 1e5, 1e5, 1e5])  
        self.norm_zeros = ((masking_values - hf['norm_params/mean'][:]) / hf['norm_params/std'][:]).astype(np.float32)

        # Charge limmiting 
        self.set_up_Q_lim = set_up_Q_lim
        self.Q_up_lim_norm = (up_Q_lim - self.means[0]) / self.stds[0]
        
        # For noise
        self.apply_add_gauss = apply_add_gauss
        self.apply_mult_gauss = apply_mult_gauss
        self.q_fraction = q_noise_fraction
        self.g_add_stds = g_add_stds / self.stds

        # load weights for energy distribution
        self.use_weights = use_weights

    def step(self, start, stop, ev_starts):
        hf = self.hf
        data_start = ev_starts[0]
        data_stop = ev_starts[-1]
        data = ((hf[f'{self.regime}/data'][data_start: data_stop] - self.means)/self.stds).astype(np.float32)
        if self.set_up_Q_lim:
            data[:, 0:1] = np.where(data[:, 0:1] > self.Q_up_lim_norm, self.Q_up_lim_norm, data[:, 0:1])
        # apply noise
        if self.apply_add_gauss:
            data = self._add_gauss(data, ev_starts)
        if self.apply_mult_gauss:
            data[:, 0] = self._mult_gauss(data[:, 0])
        data = tf.RaggedTensor.from_row_starts(values=data, row_starts=ev_starts[0:-1] - ev_starts[0])
        data = data.to_tensor(default_value=self.norm_zeros)
        mask = tf.where(tf.not_equal(data[:,:,1:2], self.norm_zeros[1:2]), 1., 0.)
        assert data.shape[-1]==5            
        data = tf.concat([data, mask], axis=-1)
        assert data.shape[-1]==6    
            
        labels = np.zeros((self.batch_size, 2))
        labels[:, 0] = (np.log10(hf[f'{self.regime}/muons_prty/individ'][start:stop]) - self.lgEmean) / self.lgEstd 

        # while check>0:
        #     try:
        #         data = tf.RaggedTensor.from_row_starts(values=data, row_starts=ev_starts[0:-1] - ev_starts[0])
        #         check = 0
        #     except Exception as e:
        #         print("Error in generator step!")
        #         print(e)
        #         print(f"{data.shape=}, {ev_starts[0]=}, {ev_starts[-1]=}")
        #         check += 1
        #         if check>10:
        #             print("Stop trying...")
        #             return None, None

        if self.use_weights:
            path_to_h5_dir = self.get_dir_name(self.path_to_h5)
            self.dist_of_weights = tf.cast(np.load(f"{path_to_h5_dir}/weights_distr_train.npy"), tf.float32)
            weights = self.get_weights_tf(labels[:, 0:1], weights_hist=self.dist_of_weights)
            return data, labels, weights
        else:
            return data, labels, tf.ones((stop - start, 1), tf.float32)

    def __call__(self):
        start = self.start
        stop = self.start + self.batch_size
        for _ in range(self.batch_num):
            ev_starts = self.hf[self.regime + '/ev_starts'][start:stop + 1]
            # # Костыль для правильной работы cnn. Есть какая-то фундаментальная проблема типа коллизии в tf
            # check=1
            # while check>0:
            #     try:
            out_data = self.step(start, stop, ev_starts)
            #        check=0
            #     except:
            #         print("Error occured")
            #         print("check num: ", check)
            #         print("Start-stop: ", start, stop)
            #         print("ev_starts shape: ", ev_starts.shape)
            #         out_data = self.step(start, stop, ev_starts)
            #         check+=1
            #         if check>50:
            #             break
            start += self.batch_size
            stop += self.batch_size
            yield out_data
            

@dataclass
class DatasetInput:
    path_to_h5: str = ""
    batch_size: int = 32
    shape: tuple = (None, 6)
    start: int = 0
    # limiting Q vals
    set_up_Q_lim: bool = True
    up_Q_lim: float = 100.
    # gauss noise
    # 0.1 ~ 1 p.e., 150 ns, 4, 4, 15 m
    apply_add_gauss: bool = False
    g_add_stds: list = None
    apply_mult_gauss: bool = False
    q_noise_fraction: float = 0.1
    use_weights: bool = False


def make_dataset(regime, ds_input: DatasetInput = DatasetInput()):
    if ds_input.g_add_stds is None:
        ds_input.g_add_stds = [1, 50, 2, 2, 5]
    bs = ds_input.batch_size
    gen_kwargs = {k: v for k, v in asdict(ds_input).items() if k not in ["shape"]}
    gen = DsGeneratorNE(regime=regime, **gen_kwargs)
    dataset = tf.data.Dataset.from_generator(
                                            gen, 
                                            output_signature=(
                                                tf.TensorSpec(shape=(bs, ds_input.shape[0], ds_input.shape[1])),
                                                tf.TensorSpec(shape=(bs, 2)),
                                                tf.TensorSpec(shape=(bs, 1))
                                                )
                                            )
    if regime == 'train':
        dataset = dataset.repeat(-1).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # output is (<the dataset>, batch_size: int, total_num_of_events: int)
    return dataset, bs, gen.num
