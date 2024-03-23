import yaml
import os
import tensorflow as tf
from AnalysisBlock.preds_funcs import make_and_save_preds

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
MODEL_NAME = "21_03_BigRNN_OnFlatSpec"

inp_dict = dict(path_to_model_dir = f"/home/albert/Baikal/NuEnergy/NNBlock/experiments/{MODEL_NAME}",
    model_regime = "best_by_train",
    ds_regime = "val",
    batch_size = 1024)
path_to_model = f"{inp_dict['path_to_model_dir']}/{inp_dict['model_regime']}"
ds_regime = inp_dict['ds_regime']

os.makedirs(f'{path_to_model}/inference', exist_ok=True)
with open(f'{path_to_model}/inference/prediction_config_on_{ds_regime}.yaml', 'w') as fp:
    yaml.dump(inp_dict, fp)

#path_to_model = f"{path_to_model_dir}/{model_regime}"
preds, labels, weights = make_and_save_preds(**inp_dict)
print("Preds are successfully made!")
print("Shapes:")
print("preds:", preds.shape)
print("labels:", labels.shape)
print("weights:", weights.shape)