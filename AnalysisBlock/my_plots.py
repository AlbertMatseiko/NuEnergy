import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm
import os

try:
    from .preds_funcs import *
except:
    from preds_funcs import *

# Plotting functions

def plot_logE_hist(pred, true, weights=None, title="HistsLog10E", path_to_save="./"):
    if weights is None:
        weights = np.array([1.] * len(pred[:, 0]))
    else:
        weights = weights[:len(pred[:, 0]), 0]

    # plotting
    true = (true[:pred.shape[0], 0], r'True $\log_{10}E$')
    pred = (pred[:, 0], r'Predicted $\log_{10}E$')
    data_list = [true, pred]
    labels = [r"True $\log_{10}E$ distibution", r"Predicted $\log_{10}E$ distibution"]
    colors = ["green", "coral"]

    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    hist_kwargs = dict(weights=weights, bins=100, density=True)
    tick_size = 20
    label_size = 30

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
    os.makedirs(f"{path_to_save}", exist_ok=True)
    plt.savefig(f"{path_to_save}/{title}.png")
    plt.close()

    return fig


def plot_hists2d(pred, true, weights=None, size: int = None,
                 title: str = "Hist2D", path_to_save: str = "./"):
    # making slices on data to have both atmospheric and astrophysics neutrino
    if size is not None:
        idxs = [i for i in range(size // 2)] + [len(pred[:, 0]) - i - 1 for i in range(size // 2)]
    else:
        idxs = [i for i in range(len(pred[:, 0]))]

    # collect energy arrays
    x_E = (true[idxs, 0], r"True $\log_{10}E$")
    y_E = (pred[idxs, 0], r"Predicted $\log_{10}E$")

    # collect error arrays
    x_s = (np.abs(y_E[0] - x_E[0]), r"True error")
    y_s = (pred[idxs, 1], r"Predicted $\sigma$")

    # collect weights
    if weights is None:
        weights = np.array([1.] * (len(x_E[0])))
    else:
        weights = weights[idxs, 0]

    # structure the objects to plot
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
        cbar = fig.colorbar(l[3], ax=a)
        cbar.ax.tick_params(labelsize=12)

    for i in range(3):
        for j in range(2):
            ax_cur = ax[i, j]
            plot_one_hist(ax_cur, data_list[i][j], hist_title=labels[i][j], add_line=(i == 0))

    os.makedirs(f"{path_to_save}", exist_ok=True)
    plt.savefig(f"{path_to_save}/{title}.png")
    plt.close()

    return fig


def plot_z(pred, true, weights=None, x_axis=np.arange(-10, 10, 0.001),  # Plot between -10 and 10 with .001 steps.
           title="Z-score distribution", path_to_save="./"):
    if weights is None:
        weights = np.array([1.] * len(pred[:, 0]))
    else:
        weights = weights[:len(pred[:, 0]), 0]

    y_true = true[:pred.shape[0], 0]
    y_pred = pred[:, 0]
    sigma = pred[:, 1]+1e-3
    values = (y_true - y_pred) / sigma
    #values = values[sigma > 0.001]

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(x_axis, norm.pdf(x_axis, 0, 1), label="Нормальное распределение, N(0,1)")
    h = plt.hist(values, bins=1000, density=True, weights=weights,
                 label="Распределение z")
    plt.grid(which='both', linestyle=':')
    plt.xlabel(r"z=$\frac{(\log_{10}E_{true}-\log_{10}E*)}{\sigma}$", fontsize=20)
    plt.ylabel(r"Density", fontsize=15)
    # plt.ylim(1e-6,0.5)
    plt.xlim(-10, 10)
    ax.minorticks_on()
    plt.legend()

    plt.title(title)
    os.makedirs(f"{path_to_save}", exist_ok=True)
    plt.savefig(f"{path_to_save}/{title}.png")
    plt.close()

    return fig


# TESTS
if __name__ == "__main__":
    print("Test is starting")
    path_to_model_dir = "/home/albert/Baikal/NuEnergy/NNBlock/experiments/SmallRNN3"
    print(load_ds_params(path_to_model_dir))

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ds_regime = "test"
    renorm = False

    preds, labels, weights = make_preds(path_to_model_dir, ds_regime=ds_regime, bath_size=1024)
    #preds, labels, weights = load_preds(path_to_model_dir, ds_regime=ds_regime, renorm=renorm)
    #print(preds.shape, labels.shape, weights.shape)
    #print(weights[0:10])

    title_postfix = f"{ds_regime}_weighted"
    path_to_model = f"{path_to_model_dir}/best_by_test"
    # fig1 = plot_logE_hist(preds, labels,  #weights,
    #                      title=f"HistsLog10E_{title_postfix}", path_to_save=f"{path_to_model}/figures")
    # fig2 = plot_hists2d(preds, labels, #weights,  #size=100000,
    #                     title=f"Hist2D_{title_postfix}", path_to_save=f"{path_to_model}/figures")

    # fig3 = plot_x(preds, labels,  #weights,
    #               title=f"X_dist_{title_postfix}", path_to_save=f"{path_to_model}/figures")

    
    MSE_w = (((preds[:,0:1]-labels[:,0:1])**2)*weights[:,0:1]).sum()/weights[:,0:1].sum()
    MAE_w = ((np.abs(preds[:,0:1]-labels[:,0:1]))*weights[:,0:1]).sum()/weights[:,0:1].sum()
    LOSS_w = (weights[:,0:1]*(((preds[:,0:1]-labels[:,0:1])**2)/(preds[:,1:2])**2 + np.log(preds[:,1:2]**2))).sum()/weights[:,0:1].sum()

    LOSS = ((preds[:,0:1]-labels[:,0:1])**2/(preds[:,1:2])**2 + np.log(preds[:,1:2]**2)).mean()
    print(MSE_w, MAE_w, LOSS_w)
    print(LOSS)

    # print(MyLoss()(labels, preds, sample_weight=weights[:,0]))
    # print(MyLoss()(labels, preds))
    # print(nlogE_MAE()(labels, preds, sample_weight=weights[:,0]))
    # print(nlogE_MAE()(labels, preds))

    print(preds.shape)

# TODO: add plot for z divergence