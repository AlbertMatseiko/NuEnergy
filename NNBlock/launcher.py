import os
import numpy as np
import yaml
from dataclasses import asdict
from train import compile_and_train
from load_from_config import dataset_from_config, model_from_config, train_input_from_config


# function for counting parameters of model
def count_params(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    return trainableParams, nonTrainableParams


def launch_exp(EXPERIMENT_NAME: str, MODEL_FUNC_NAME: str = "TwoTapesModel",
               enc_yml_name: str = "EncoderConfig",
               energy_yml_name: str = "EnergyBlockConfig",
               sigma_yml_name: str = "SigmaBlockConfig",
               dataset_yml_name: str = "DatasetConfig",
               compile_train_yml_name: str = "CompileAndTrainConfig",
               DESCRIPTION: str = ""):
    # Make dir for experiment
    path_to_save = f"./experiments/{EXPERIMENT_NAME}"
    os.makedirs(path_to_save, exist_ok=False)

    # Save short description to model directory
    print(f"{DESCRIPTION}", file=open(f'{path_to_save}/description.txt', 'w'))

    # Create model
    model, model_inp = model_from_config(MODEL_FUNC_NAME,
                                         enc_yml_name,
                                         energy_yml_name,
                                         sigma_yml_name)
    model.build(input_shape=(None, None, 6))

    # Make logs of model config
    with open(f'{path_to_save}/model_hyperparams.yaml', 'w') as file:
        yaml.dump(asdict(model_inp), file)
    print(f"Model architectue is: {MODEL_FUNC_NAME}, \
            Number of params: {count_params(model)}", file=open(f'{path_to_save}/summary.txt', 'w'))

    # Load datasets with info about batchsize and total events num
    train_ds_with_info, dataset_input = dataset_from_config('train', yml_name=dataset_yml_name)
    test_ds_with_info, _ = dataset_from_config('test', yml_name=dataset_yml_name)

    # Make logs of dataset config
    with open(f'{path_to_save}/dataset_params.yaml', 'w') as file:
        yaml.dump(asdict(dataset_input), file)

    # Start the fit! Logging of train config is inside!
    history = compile_and_train(model,
                                path_to_save=path_to_save,
                                train_ds_with_info=train_ds_with_info,
                                test_ds_with_info=test_ds_with_info,
                                input_args=train_input_from_config(yml_name=compile_train_yml_name))
    return history