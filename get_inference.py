from NNBlock.preds_funcs import make_preds, load_preds
from AnalysisBlock.my_plots import plot_logE_hist, plot_hists2d, plot_x
from NNBlock.nn.losses import MyLoss
from NNBlock.nn.metrics import nlogE_MAE, nlogE_MSE
import yaml
import os
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# INPUTS
path_to_model_dir = "/home/albert/Baikal/NuEnergy/NNBlock/experiments/BigRNN_NoNoise"
model_regime = "best_by_train"
ds_regime = "val"
renorm = True
make_new_preds = False
batch_size=1024
plot_with_weights = True

# MAIN PART
path_to_model = f"{path_to_model_dir}/{model_regime}"

inp_dict = dict(path_to_model_dir = path_to_model_dir,
    model_regime = model_regime,
    ds_regime = ds_regime,
    renorm = renorm,
    make_new_preds = make_new_preds,
    batch_size = batch_size,
    plot_with_weights=plot_with_weights)

os.makedirs(f'{path_to_model}/inference', exist_ok=True)
with open(f'{path_to_model}/inference/inference_config_on_{ds_regime}.yaml', 'w') as fp:
    yaml.dump(inp_dict, fp)

# make or load predictions
if make_new_preds:
    preds, labels, weights = make_preds(path_to_model_dir, model_regime=model_regime, ds_regime=ds_regime, bath_size=batch_size)
preds, labels, weights = load_preds(path_to_model_dir, model_regime=model_regime, ds_regime=ds_regime, renorm=renorm)

if ds_regime=="val":
    num = 200_000
    preds = np.concatenate([preds[:num], preds[-num:]], axis=0)
    labels = np.concatenate([labels[:num], labels[-num:]], axis=0)
    weights = np.concatenate([weights[:num], weights[-num:]], axis=0)

    
# evaluate
res = dict(
    loss_w = float(MyLoss()(labels, preds, sample_weight=weights).numpy()),
    loss = float(MyLoss()(labels, preds).numpy()),
    mae_w = float(nlogE_MAE()(labels, preds, sample_weight=weights).numpy()),
    mae = float(nlogE_MAE()(labels, preds).numpy()),
    mse_w = float(nlogE_MSE()(labels, preds, sample_weight=weights).numpy()),
    mse = float(nlogE_MSE()(labels, preds).numpy())
)
print(res)
with open(f'{path_to_model}/inference/evaluate_result_on_{ds_regime}.yaml', 'w') as fp:
    yaml.dump(res, fp)

# generate titles, set plot  weights
if plot_with_weights:
    title_postfix = f"{ds_regime}_weighted"
    plot_w = weights
else:
    plot_w = np.ones((preds.shape[0],1))
    title_postfix = f"{ds_regime}"

# plotting
fig1 = plot_logE_hist(preds, labels, plot_w,
                      title=f"HistsLog10E_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig2 = plot_hists2d(preds, labels, plot_w,
                    title=f"Hist2D_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig3 = plot_x(preds, labels, plot_w,
               title=f"z_dist_{title_postfix}", path_to_save=f"{path_to_model}/figures")
