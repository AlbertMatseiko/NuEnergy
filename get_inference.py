from AnalysisBlock.preds_funcs import load_preds
from AnalysisBlock.my_plots import plot_logE_hist, plot_hists2d, plot_z
from AnalysisBlock.metric_funcs import my_loss, mae, mse
import yaml
import os
import numpy as np

# INPUTS
MODEL_NAME = "25_03_MiddleCNN_OnFlatSpec"

inp_dict = dict(path_to_model_dir = f"/home/albert/Baikal/NuEnergy/NNBlock/experiments/{MODEL_NAME}",
    model_regime = "best_by_test",
    ds_regime = "val",
    renorm = True)

# MAIN
path_to_model = f"{inp_dict['path_to_model_dir']}/{inp_dict['model_regime']}"
ds_regime = inp_dict['ds_regime']

os.makedirs(f'{path_to_model}/inference', exist_ok=True)
with open(f'{path_to_model}/inference/inference_config_on_{ds_regime}.yaml', 'w') as fp:
    yaml.dump(inp_dict, fp)

# loading preds
try:
    preds, labels, weights = load_preds(**inp_dict)
except:
    print("Can not load preds from h5! Probably, they were not made!")

# EVALUATE
res = dict(
    loss_w = float(my_loss(labels, preds, sample_weight=weights)),
    loss = float(my_loss(labels, preds)),
    mae_w = float(mae(labels, preds, sample_weight=weights)),
    mae = float(mae(labels, preds)),
    mse_w = float(mse(labels, preds, sample_weight=weights)),
    mse = float(mse(labels, preds))
)
print(res)
with open(f'{path_to_model}/inference/evaluate_result_on_{ds_regime}.yaml', 'w') as fp:
    yaml.dump(res, fp)
    
# PLOTTING
# without weights
title_postfix = f"{ds_regime}"
fig1 = plot_logE_hist(preds, labels,
                      title=f"HistsLog10E_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig2 = plot_hists2d(preds, labels,
                    title=f"Hist2D_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig3 = plot_z(preds, labels,
               title=f"z_dist_{title_postfix}", path_to_save=f"{path_to_model}/figures")

# with weights
title_postfix = f"{ds_regime}_weighted"
fig1 = plot_logE_hist(preds, labels, weights,
                      title=f"HistsLog10E_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig2 = plot_hists2d(preds, labels, weights,
                    title=f"Hist2D_{title_postfix}", path_to_save=f"{path_to_model}/figures")
fig3 = plot_z(preds, labels, weights,
               title=f"z_dist_{title_postfix}", path_to_save=f"{path_to_model}/figures")
