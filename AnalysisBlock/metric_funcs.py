import numpy as np


def my_loss(y_true, y_pred, sample_weight=None, eps=1e-3):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))

    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    y_sigma_pred = y_pred[:, 1:2]  
    loss = np.sum(np.multiply(np.log((y_sigma_pred + eps) ** 2) + (y_e_pred - y_e_true) ** 2 /
                                        ((y_sigma_pred + eps) ** 2),
                                        sample_weight)
                            ) / np.sum(sample_weight)
    return loss


def mse(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))
    
    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    
    return np.sum((y_e_true-y_e_pred)**2 * sample_weight)/sample_weight.sum()


def mae(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))
    
    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    
    return np.sum(np.abs(y_e_true-y_e_pred) * sample_weight)/sample_weight.sum()