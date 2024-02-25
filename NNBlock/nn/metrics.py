import tensorflow as tf


# METRICS FOR NORMED log10E values
class nlogE_MAE(tf.keras.metrics.Metric):
    def __init__(self, name='nlogE_MAE', weighted: bool = True, **kwargs):
        if weighted:
            name += "_weighted"
        super(nlogE_MAE, self).__init__(name=name, **kwargs)
        self.MAE = tf.keras.metrics.MeanAbsoluteError()
        self.weighted = weighted

    def update_state(self, y_true, y_pred, sample_weight=1.):
        if self.weighted:
            self.MAE.update_state(y_true[:, 0:1], y_pred[:, 0:1], sample_weight=sample_weight)
        else:
            self.MAE.update_state(y_true[:, 0:1], y_pred[:, 0:1])

    def reset_state(self):
        self.MAE.reset_state()

    def result(self):
        return self.MAE.result()


class nlogE_MSE(tf.keras.metrics.Metric):
    def __init__(self, name='nlogE_MSE', weighted: bool = True, **kwargs):
        if weighted:
            name += "_weighted"
        super(nlogE_MSE, self).__init__(name=name, **kwargs)
        self.MSE = tf.keras.metrics.MeanSquaredError()
        self.weighted = weighted

    def update_state(self, y_true, y_pred, sample_weight=1.):
        if self.weighted:
            self.MSE.update_state(y_true[:, 0:1], y_pred[:, 0:1], sample_weight)
        else:
            self.MSE.update_state(y_true[:, 0:1], y_pred[:, 0:1])

    def reset_state(self):
        self.MSE.reset_state()

    def result(self):
        return self.MSE.result()


class nSigmaMSE(tf.keras.metrics.Metric):
    def __init__(self, name='nSigmaMSE', weighted: bool = True, **kwargs):
        if weighted:
            name += "_weighted"
        super(nSigmaMSE, self).__init__(name=name, **kwargs)
        self.MSE = tf.keras.metrics.MeanSquaredError()
        self.weighted = weighted

    def update_state(self, y_true, y_pred, sample_weight=None):
        nlogE_true = y_true[:, 0:1]
        nlogE_pred = y_pred[:, 0:1]
        nsigma_true = tf.abs(nlogE_true - nlogE_pred)
        nsigma_pred = y_pred[:, 1:2]
        if self.weighted:
            self.MSE.update_state(nsigma_true, nsigma_pred, sample_weight)
        else:
            self.MSE.update_state(nsigma_true, nsigma_pred)

    def reset_state(self):
        self.MSE.reset_state()

    def result(self):
        return self.MSE.result()


class nSigmaMAE(tf.keras.metrics.Metric):
    def __init__(self, name='nSigmaMAE', weighted: bool = True, **kwargs):
        if weighted:
            name += "_weighted"
        super(nSigmaMAE, self).__init__(name=name, **kwargs)
        self.MAE = tf.keras.metrics.MeanAbsoluteError()
        self.weighted = weighted

    def update_state(self, y_true, y_pred, sample_weight=None):
        nlogE_true = y_true[:, 0:1]
        nlogE_pred = y_pred[:, 0:1]
        nsigma_true = tf.abs(nlogE_true - nlogE_pred)
        nsigma_pred = y_pred[:, 1:2]
        if self.weighted:
            self.MAE.update_state(nsigma_true, nsigma_pred, sample_weight)
        else:
            self.MAE.update_state(nsigma_true, nsigma_pred)

    def reset_state(self):
        self.MAE.reset_state()

    def result(self):
        return self.MAE.result()


# METRICS FOR log10E values
Mean, Std = 3.782758, 1.0855767


class logE_MAE(tf.keras.metrics.Metric):
    def __init__(self, name='logE_MAE', **kwargs):
        super(logE_MAE, self).__init__(name=name, **kwargs)
        self.MAE = tf.keras.metrics.MeanAbsoluteError()
        self.mean = Mean
        self.std = Std

    def update_state(self, y_true, y_pred, sample_weight=None):
        logE_true = y_true[:, 0:1] * self.std + self.mean
        logE_pred = y_pred[:, 0:1] * self.std + self.mean
        self.MAE.update_state(logE_true, logE_pred, sample_weight)

    def reset_state(self):
        self.MAE.reset_state()

    def result(self):
        return self.MAE.result()


class logE_MSE(tf.keras.metrics.Metric):
    def __init__(self, name='logE_MSE', **kwargs):
        super(logE_MSE, self).__init__(name=name, **kwargs)
        self.MSE = tf.keras.metrics.MeanSquaredError()
        self.mean = tf.cast(Mean, tf.float32)
        self.std = tf.cast(Std, tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        logE_true = y_true[:, 0:1] * self.std + self.mean
        logE_pred = y_pred[:, 0:1] * self.std + self.mean
        # logE_true = y_true * self.std + self.mean
        # logE_pred = y_pred * self.std + self.mean
        self.MSE.update_state(logE_true, logE_pred, sample_weight)

    def reset_state(self):
        self.MSE.reset_state()

    def result(self):
        return self.MSE.result()


class SigmaMSE(tf.keras.metrics.Metric):
    def __init__(self, name='SigmaMSE', **kwargs):
        super(SigmaMSE, self).__init__(name=name, **kwargs)
        self.MSE = tf.keras.metrics.MeanSquaredError()
        self.mean = Mean
        self.std = Std

    def update_state(self, y_true, y_pred, sample_weight=None):
        logE_true = y_true[:, 0:1] * self.std + self.mean
        logE_pred = y_pred[:, 0:1] * self.std + self.mean
        sigma_true = tf.abs(logE_true - logE_pred)
        sigma_pred = y_pred[:, 1:2] * self.std
        self.MSE.update_state(sigma_true, sigma_pred, sample_weight)

    def reset_state(self):
        self.MSE.reset_state()

    def result(self):
        return self.MSE.result()


class SigmaMAE(tf.keras.metrics.Metric):
    def __init__(self, name='SigmaMAE', **kwargs):
        super(SigmaMAE, self).__init__(name=name, **kwargs)
        self.MAE = tf.keras.metrics.MeanAbsoluteError()
        self.mean = Mean
        self.std = Std

    def update_state(self, y_true, y_pred, sample_weight=None):
        logE_true = y_true[:, 0:1] * self.std + self.mean
        logE_pred = y_pred[:, 0:1] * self.std + self.mean
        sigma_true = tf.abs(logE_true - logE_pred)
        sigma_pred = y_pred[:, 1:2] * self.std
        self.MAE.update_state(sigma_true, sigma_pred, sample_weight)

    def reset_state(self):
        self.MAE.reset_state()

    def result(self):
        return self.MAE.result()


class EnTruePredRatio(tf.keras.metrics.Metric):
    def __init__(self, name='EnTruePredRatio', **kwargs):
        super(EnTruePredRatio, self).__init__(name=name, **kwargs)
        self.mean = Mean
        self.std = Std
        self.ratios = tf.constant(0., shape=(1,))

    def update_state(self, y_true, y_pred, sample_weight=None):
        logE_true = y_true[:, 0:1] * self.std + self.mean
        logE_pred = y_pred[:, 0:1] * self.std + self.mean
        self.ratios = tf.keras.layers.Concatenate(axis=0)([self.ratios, logE_true - logE_pred])

    def reset_state(self):
        self.ratios = tf.constant(0., shape=(1,))

    def result(self):
        return tf.pow(10., tf.reduce_mean(self.ratios))


# TESTING DRAFT
if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    X = tf.random.uniform([5, 1], -1, 1, dtype=tf.float32, seed=0)
    print(X)
    Y = tf.random.uniform([5, 1], -0.1, 0.1, dtype=tf.float32, seed=0)
    print(Y)
    print(tf.reduce_mean(tf.abs(X - Y)))
    W = tf.cast([[1], [100], [3], [4], [5]], tf.float32)
    print(W.shape)

    mae = tf.keras.metrics.MeanAbsoluteError()
    m = nlogE_MAE(weighted=False)
    m_w = nlogE_MAE(weighted=True)

    m.update_state(Y, X)
    m_w.update_state(Y, X, sample_weight=W)
    mae.update_state(Y, X, sample_weight=W)
    print(m.result())
    print(m_w.result())
    print(mae.result())

    print(m_w.MAE.result())
    m_w.reset_state()
    m_w.MAE.update_state(Y, X, sample_weight=W)
    print(m_w.MAE.result())
