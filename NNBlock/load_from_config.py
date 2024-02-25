"""
Funcs for loading <structures> by their configs, that are described in yml_configs directory.
"""
import yaml
from pathlib import Path

from dataset_from_h5 import DatasetInput, make_dataset
from nn.custom_layers import EncoderBlockInput, DenseRegressionInput
import models
from train import CompileAndTrainInput


def dataset_from_config(regime, yml_name="DatasetConfig"):
    ds_path = Path(__file__).parent.absolute() / f"yml_configs/{yml_name}.yaml"
    ds_dict = yaml.safe_load(Path(ds_path).read_text())
    ds_inp = DatasetInput(**ds_dict)
    return make_dataset(regime, ds_input=ds_inp), ds_inp


def model_from_config(MODEL_FUNC_NAME,
                      enc_yml_name="EncoderConfig",
                      energy_yml_name="EnergyBlockConfig",
                      sigma_yml_name="SigmaBlockConfig"):
    # load encoder config
    enc_path = Path(__file__).parent.absolute() / f"yml_configs/{enc_yml_name}.yaml"
    enc_dict = yaml.safe_load(Path(enc_path).read_text())
    enc_inp = EncoderBlockInput(**enc_dict)

    # load energy block config
    energy_path = Path(__file__).parent.absolute() / f"yml_configs/{energy_yml_name}.yaml"
    energy_dict = yaml.safe_load(Path(energy_path).read_text())
    energy_inp = DenseRegressionInput(**energy_dict)

    # load error block config
    sigma_path = Path(__file__).parent.absolute() / f"yml_configs/{sigma_yml_name}.yaml"
    sigma_dict = yaml.safe_load(Path(sigma_path).read_text())
    if sigma_dict['dense_blocks'] is not None:
        sigma_inp = DenseRegressionInput(**sigma_dict)
    else:
        sigma_inp = None

    # create the model
    inp = models.TwoTapesModelInput(encoder_inp=enc_inp, energy_inp=energy_inp, sigma_inp=sigma_inp)
    return getattr(models, MODEL_FUNC_NAME)(inp), inp


def train_input_from_config(yml_name="CompileAndTrainConfig"):
    cfg_path = Path(__file__).parent.absolute() / f"yml_configs/{yml_name}.yaml"
    cfg_dict = yaml.safe_load(Path(cfg_path).read_text())
    return CompileAndTrainInput(**cfg_dict)
