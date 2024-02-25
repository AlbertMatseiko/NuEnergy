import numpy as np
import h5py as h5
import yaml
from pathlib import Path
from h5_info.funcs import collect_info


def step3(config_dict, h5_name=None):
    if h5_name is None:
        h5_name = config_dict["h5_name"]
    step_name = config_dict["step_name"]
    postfix = step_name[5:]
    if config_dict['norm_log_energy']:
        postfix += '_normlog10E'
    path_to_output = Path(__file__).parent.absolute() / "data" / h5_name
    path_to_input_h5 = path_to_output / f"{step_name}.h5"
    path_to_output_h5 = path_to_output / f"step3{postfix}_normed.h5"
    path_to_log = f"{path_to_output}/step3{postfix}_logs.txt"
    log_file = open(path_to_log, 'w')
    with h5.File(path_to_input_h5, 'r') as hf:
        with h5.File(path_to_output_h5, 'w') as hfout:
            data_train = np.array(hf["train/data"][:], dtype=np.float64)
            mean, std = np.mean(data_train, axis=0, dtype=np.float64), np.std(data_train, axis=0, dtype=np.float64)
            print(f"Means: {mean} \nStds: {std} \n{mean.dtype}", file=open(path_to_log, 'a'))
            data_train = 0
            hfout.create_dataset(f"norm_params/mean", data=mean)
            hfout.create_dataset(f"norm_params/std", data=std)

            for regime in list(hf.keys()):
                data_normed = np.array((hf[f"{regime}/data"][:] - mean) / std, dtype=np.float32)
                hfout.create_dataset(f"{regime}/data", data=data_normed)

                keys_list = [k for k in list(hf[regime].keys()) if k != 'data']
                for key in keys_list:
                    hfout.create_dataset(f"{regime}/{key}", data=hf[f"{regime}/{key}"][:])

            if config_dict['norm_log_energy']:
                log10e = np.array(hf["train/log10Emu"][:], dtype=np.float64)
                log10e_mean, log10e_std = log10e.mean(axis=0, dtype=np.float64), log10e.std(axis=0, dtype=np.float64)
                print(f"Means for log10Emu: {log10e_mean} \nStds for log10Emu: {log10e_std} \n{log10e_std.dtype}",
                      file=open(path_to_log, 'a'))
                hfout.create_dataset(f"norm_params/log10Emu_mean", data=log10e_mean)
                hfout.create_dataset(f"norm_params/log10Emu_std", data=log10e_std)
                for regime in list(hf.keys()):
                    log10e_normed = np.array((hf[f"{regime}/log10Emu"][:] - log10e_mean) / log10e_std,
                                             dtype=np.float32)
                    hfout.create_dataset(f"{regime}/log10Emu_norm", data=log10e_normed)

    collect_info(path_to_output_h5, path_to_output, name=f"Step3{postfix}FileStructure")
    log_file.close()

    return f"step3{postfix}_normed"


if __name__ == "__main__":
    path_to_yml = Path(__file__).parent.absolute() / 'step3.yml'
    config_dict = yaml.safe_load(Path(path_to_yml).read_text())
    step3(config_dict)