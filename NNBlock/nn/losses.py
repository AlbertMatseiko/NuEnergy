from abc import ABC
import tensorflow as tf


class MyMse(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))
        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply((y_e_pred - y_e_true) ** 2, weights)) / tf.reduce_sum(weights)
        return loss
    
class MyMseTanh(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))
        y_e_true = tf.tanh(tf.cast(y_true[:, 0:1], tf.float32))  # these are NORMED LOG10 Energies
        y_e_pred = tf.tanh(tf.cast(y_pred[:, 0:1], tf.float32))

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply((y_e_pred - y_e_true) ** 2, weights)) / tf.reduce_sum(weights)
        return loss

class MyMseForSigma(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))
        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)
        
        y_sigma_pred = tf.cast(y_pred[:, 1:2], tf.float32)
        y_sigma_true = tf.abs(y_e_true - y_e_pred)

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply((y_sigma_pred - y_sigma_true) ** 2, weights)) / tf.reduce_sum(weights)
        return loss

class MyMae(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))
        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply(tf.abs(y_e_pred - y_e_true), weights)) / tf.reduce_sum(weights)
        return loss

class MyMaeForSigma(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))
        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)
        
        y_sigma_pred = tf.cast(y_pred[:, 1:2], tf.float32)
        y_sigma_true = tf.abs(y_e_true - y_e_pred)

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply(tf.abs(y_sigma_pred - y_sigma_true), weights)) / tf.reduce_sum(weights)
        return loss


class MyLoss(tf.keras.losses.Loss, ABC):
    def __init__(self, eps=1e-5):
        self.name = "MyLoss"
        super().__init__(name=self.name)
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = tf.ones((y_true.shape[0],1))
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))

        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)
        y_sigma_pred = tf.cast(y_pred[:, 1:2], tf.float32)
        weights = tf.cast(sample_weight, tf.float32)
        loss = tf.reduce_sum(tf.multiply(tf.math.log(y_sigma_pred ** 2 + self.eps) + (y_e_pred - y_e_true) ** 2 /
                                         (y_sigma_pred**2 + self.eps),
                                         weights)
                             ) / tf.reduce_sum(weights)
        return loss
