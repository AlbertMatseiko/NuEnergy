import numpy as np
import h5py as h5
import time
import yaml
from pathlib import Path

from h5_info.funcs import collect_info


# generate postfix to step2 h5 name from config
def generate_postfix(config_dict):
    nu_regime = config_dict['neutrino']
    if nu_regime == 'all':
        ratio = config_dict['atm_e2_ratio_in_train_test']
    else:
        ratio = ''
    num_train = config_dict['num_in_train']
    num_test = config_dict['num_in_test']
    return f"_{nu_regime}_{ratio[0]}_{ratio[1]}_train{num_train}_test{num_test}"


# routine to shuffle flat data
def shuffle_data(permutation, data, ev_starts):
    starts = ev_starts[:-1]
    ends = ev_starts[1:]
    data_new = np.zeros(data.shape, dtype=np.float32)
    ev_starts_new = np.zeros(ev_starts.shape, dtype=int)

    current_data_new_index = 0
    for j, idx in enumerate(permutation):
        start, stop = starts[idx], ends[idx]
        length = stop - start
        data_new[current_data_new_index:current_data_new_index + length] = data[start:stop]
        current_data_new_index += length
        ev_starts_new[j + 1] = ev_starts_new[j] + length
    return data_new, ev_starts_new


def get_indxs_dict(hf, config_dict, path_to_log):
    len_train = config_dict[f'num_in_train']
    len_test = config_dict[f'num_in_test']
    ratio = config_dict['atm_e2_ratio_in_train_test'][0] / config_dict['atm_e2_ratio_in_train_test'][1]


    if config_dict['neutrino'] == 'all':
        idxs = {'nuatm': {}, 'nu2': {}}
        # get train
        len_train_nuatm = int(ratio / (1 + ratio) * len_train)
        len_train_nu2 = len_train - len_train_nuatm
        idxs['nuatm']['train'] = list(range(0, len_train_nuatm))
        idxs['nu2']['train'] = list(range(0, len_train_nu2))

        # get test
        len_test_nuatm = int(ratio / (1 + ratio) * len_test)
        len_test_nu2 = len_test - len_test_nuatm
        idxs['nuatm']['test'] = list(range(len_train_nuatm, len_train_nuatm + len_test_nuatm))
        idxs['nu2']['test'] = list(range(len_train_nu2, len_train_nu2 + len_test_nu2))

        # get val
        idxs['nuatm']['val'] = list(range(len_train_nuatm + len_test_nuatm, hf[f"nuatm/ev_ids"].shape[0]))
        idxs['nu2']['val'] = list(range(len_train_nu2 + len_test_nu2, hf[f"nu2/ev_ids"].shape[0]))
    else:
        particle = config_dict['neutrino']
        idxs = {particle: {}}
        # get train
        idxs[particle]['train'] = list(range(0, len_train))
        # get test
        idxs[particle]['test'] = list(range(len_train, len_train + len_test))
        # get val
        idxs[particle]['val'] = list(range(len_train + len_test, hf[f"{particle}/ev_ids"].shape[0]))

    if path_to_log is not None:
        for p, d in idxs.items():
            for regime in d.keys():
                print(f"{p} {regime} number: {len(d[regime])}\n",
                      file=open(path_to_log, 'a'))
    return idxs


def step2(config_dict, filtered_h5_name=None):
    '''
    Routine to split filtered h5 dataset in train, test and val datasets.
    '''
    if filtered_h5_name is None:
        filtered_h5_name = config_dict['filtered_h5_name']
    postfix = generate_postfix(config_dict)

    time0 = time.time()
    path_to_output = Path(__file__).parent.absolute() / "data" / filtered_h5_name
    path_to_filtered_h5 = path_to_output / "step1.h5"
    path_to_output_h5 = path_to_output / f"step2{postfix}.h5"

    # make dir to process
    # touch log file
    path_to_log = f"{path_to_output}/step2{postfix}_logs.txt"

    with h5.File(path_to_filtered_h5, 'r') as hf:
        with h5.File(path_to_output_h5, 'w') as hfout:

            # prepare indxs for each particle to take in new h5
            ev_idxs_dict = get_indxs_dict(hf, config_dict, path_to_log=path_to_log)

            # make datasets by their events idxs
            for regime in ['train', 'test', 'val']:
                print(f"Regime: {regime}", file=open(path_to_log, 'a'))

                # separately work with flat data
                data_fields = ["data", "ev_starts"]
                particle_keys = list(ev_idxs_dict.keys())
                data_list = []
                ev_starts_list = []
                print(f"  Data and EvStarts", file=open(path_to_log, 'a'))
                for particle in particle_keys:
                    idxs = ev_idxs_dict[particle][regime]
                    starts = hf[f"{particle}/ev_starts"][idxs + [idxs[-1] + 1]]
                    start, end = starts[0], starts[-1]
                    data_list.append(hf[f"{particle}/data"][start:end])
                    ev_starts_list.append(starts - start)
                    print(f"  {particle} is Done", file=open(path_to_log, 'a'))
                data_new = np.concatenate(data_list, axis=0)
                print(f"  data recalculated", file=open(path_to_log, 'a'))
                # shift nu hits' starts in order to concat mu and nu flat data correctly
                for i in range(1, len(ev_starts_list)):
                    ev_starts_list[i] = ev_starts_list[i][1:] + ev_starts_list[i - 1][-1]
                ev_starts_new = np.concatenate(ev_starts_list, axis=0)
                print(f"  ev_starts recalculated", file=open(path_to_log, 'a'))
                print(f"    Total hits: {data_new.shape[0]}; Total events: {ev_starts_new.shape[0]}",
                      file=open(path_to_log, 'a'))
                assert ev_starts_new[-1] == data_new.shape[0]
                # shuffle if train
                if regime == 'train':
                    # make permuted indexes of events for train
                    permutation = np.random.permutation(ev_starts_new.shape[0] - 1)
                    data_new, ev_starts_new = shuffle_data(permutation, data_new, ev_starts_new)
                    print(
                        f"    Permutation is done. Total hits: {data_new.shape[0]}; Total events: {ev_starts_new.shape[0]}",
                        file=open(path_to_log, 'a'))
                hfout.create_dataset(f"{regime}/data", data=data_new)
                hfout.create_dataset(f"{regime}/ev_starts", data=ev_starts_new)

                # separately work with not flat info
                events_fields = [f for f in list(hf[f"nuatm"].keys()) if f not in data_fields]
                print(f"  Other Keys", file=open(path_to_log, 'a'))
                for field in events_fields:
                    particle_keys = list(ev_idxs_dict.keys())
                    list_to_concat = []
                    print(f"  Key: {field}", file=open(path_to_log, 'a'))
                    for particle in particle_keys:
                        idxs = ev_idxs_dict[particle][regime]
                        array = hf[f"{particle}/{field}"][idxs]
                        list_to_concat.append(array)
                        print(f"    {particle} is Done", file=open(path_to_log, 'a'))
                    new_array = np.concatenate(list_to_concat, axis=0)
                    print(f"    output shape: {new_array.shape}", file=open(path_to_log, 'a'))
                    # shuffle if train
                    if regime == 'train':
                        new_array = new_array[permutation]
                        print(f"  output shape after permutation: {new_array.shape}", file=open(path_to_log, 'a'))
                    hfout.create_dataset(f"{regime}/{field}", data=new_array)
    collect_info(path_to_output_h5, path_to_output, name=f"Step2{postfix}FileStructure")
    print(f"Totally passed = {time.time() - time0}")

    return f"step2{postfix}"


if __name__ == "__main__":
    path_to_yml = Path(__file__).parent.absolute() / 'step2.yml'
    config_dict = yaml.safe_load(Path(path_to_yml).read_text())
    step2(config_dict)
