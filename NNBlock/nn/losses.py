from abc import ABC
import tensorflow as tf


class MyMSE(tf.keras.losses.Loss, ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, sample_weight=1.):
        y_e_true = tf.cast(y_true[:, 0], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0], tf.float32)

        weights = sample_weight
        loss = tf.reduce_sum(tf.multiply((y_e_pred - y_e_true) ** 2, weights)) / tf.reduce_sum(weights)
        return loss


class MyLoss(tf.keras.losses.Loss, ABC):
    def __init__(self, eps=1e-3):
        self.name = "MyLoss"
        super().__init__(name=self.name)
        self.eps = eps

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = [[1.]]*y_true.shape[0]
        sample_weight = tf.reshape(sample_weight, (sample_weight.shape[0],1))

        y_e_true = tf.cast(y_true[:, 0:1], tf.float32)  # these are NORMED LOG10 Energies
        y_e_pred = tf.cast(y_pred[:, 0:1], tf.float32)
        y_sigma_pred = tf.cast(y_pred[:, 1:2], tf.float32)
        weights = tf.cast(sample_weight, tf.float32)
        loss = tf.reduce_sum(tf.multiply(tf.math.log((y_sigma_pred + self.eps) ** 2) + (y_e_pred - y_e_true) ** 2 /
                                         ((y_sigma_pred + self.eps) ** 2),
                                         weights)
                             ) / tf.reduce_sum(weights)
        return loss
