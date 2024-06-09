import numpy as np


def my_loss(y_true, y_pred, sample_weight=None, eps=1e-3):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))

    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    y_sigma_pred = y_pred[:, 1:2]  
    loss_i = np.multiply(np.log(y_sigma_pred ** 2 +eps) + (y_e_pred - y_e_true) ** 2 /
                                        (y_sigma_pred**2 + eps),
                                        sample_weight)
    err = loss_i.std()/np.sqrt(np.sum(sample_weight))
    loss = np.sum(loss_i) / np.sum(sample_weight)
    return loss, err


def mse(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))
    
    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    res_i = (y_e_true-y_e_pred)**2 * sample_weight
    err = res_i.std()/np.sqrt(np.sum(sample_weight))
    res = np.sum(res_i)/sample_weight.sum()
    return res, err


def mae(y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones((y_true.shape[0],1))
    sample_weight = np.reshape(sample_weight, (sample_weight.shape[0],1))
    
    y_e_true = y_true[:, 0:1], # these are NORMED LOG10 Energies
    y_e_pred = y_pred[:, 0:1]
    
    res_i = np.abs(y_e_true-y_e_pred) * sample_weight
    err = res_i.std()/np.sqrt(np.sum(sample_weight))
    res = np.sum(res_i)/sample_weight.sum()
    return res, err