import numpy as np
import h5py as h5
import os
import sys
import time
import yaml
from pathlib import Path

from h5_info.funcs import get_structure_h5, get_h5_name, collect_info


def generate_postfix(filters):
    out = ""
    if filters['only_signal']:
        out += "_signal"
    if filters['filter_doubles']:
        out += "_nodoubles"
    for key, name_in_postfix in [('hits', 'H'), ('Q', 'Q'), ('strings', 'S'),
                                 ('E_prime', 'Enu'),
                                 ('E_indmu', 'Emu')]:
        start, end = filters[key]
        if end > 10 ** 8:
            end = ""
        else:
            end = f"to{end}"
        if start > 0:
            out += f"_{name_in_postfix}{start}{end}"
    return out


def step1(config_dict):
    '''
    Routine to convert pure MC flat h5 file with parts into filtered h5 dataset without parts keys.
    The dataset is designed for MuNu task only.
    '''

    path_to_h5 = config_dict['path_to_h5']
    bad_parts = config_dict['bad_parts']
    fields_to_map = config_dict['fields_to_map']
    filters = config_dict['filters']
    particle_keys = config_dict['pk_list']

    time0 = time.time()
    h5_name = get_h5_name(path_to_h5)
    postfix = generate_postfix(filters)

    # make dir to process
    path_to_output = f"{str(Path(__file__).parent.absolute())}/data/{h5_name}{postfix}"
    os.makedirs(path_to_output, exist_ok=True)
    # touch log file
    path_to_log = f"{path_to_output}/step1_logs.txt"

    # write info about initial file
    collect_info(path_to_h5, path_to_output, name="OriginFileStructure")

    with h5.File(path_to_h5, 'r') as hf:
        with h5.File(f"{path_to_output}/step1.h5", 'w') as hfout:
            for pk in particle_keys:
                # filter bad parts
                parts_list = [p for p in list(hf[f"{pk}/ev_ids"].keys()) if p[5:9] not in bad_parts[pk]]
                print(f"Particle: {pk}, num of parts: {len(parts_list)}", file=open(path_to_log, 'a'))
                for number, (origin, image) in enumerate(fields_to_map.items()):
                    print(f"  Field: {origin}", file=open(path_to_log, 'a'))
                    # 'raw' key should be the first
                    assert (number == 0 and origin == 'raw') or (number != 0 and origin != 'raw')
                    data_to_copy = [0] * len(parts_list)
                    # flat data goes specially
                    if origin == "raw":
                        # Initialize lists of variables for each part.
                        ev_starts_list = [0] * len(parts_list)
                        num_un_strings_list = [0] * len(parts_list)
                        # idxs of hits and events that passed filters
                        idxs_hits = [0] * len(parts_list)
                        idxs_events = [0] * len(parts_list)
                    # iterate over parts (list is the same for many fields)
                    for i, part in enumerate(parts_list):
                        # flat data goes specially
                        time1 = time.time()
                        num_ev = hf[f"{pk}/raw/ev_starts/{part}/data"].shape[0] - 1
                        num_hits = hf[f"{pk}/raw/data/{part}/data"].shape[0]
                        print(f"    #{i} {part}, events: {num_ev}, hits: {num_hits}", file=open(path_to_log, 'a'))
                        if origin == "raw":
                            # filter data
                            starts = hf[f"{pk}/raw/ev_starts/{part}/data"][:]
                            ## make hits mask to apply to flat data
                            Q = hf[f"{pk}/raw/data/{part}/data"][:, 0]
                            mask_Q = np.where(Q >= filters['Q'][0], True, False) * np.where(Q <= filters['Q'][1], True,
                                                                                            False)
                            if filters['only_signal']:
                                labels = hf[f"{pk}/raw/labels/{part}/data"][:]
                                mask_signal = np.where(labels == 0, False, True)
                            else:
                                mask_signal = True
                            mask_hits = mask_Q * mask_signal

                            # recalculate num_un_strings
                            tr_chs = hf[f"{pk}/raw/channels/{part}/data"][:]
                            tr_chs = np.where(mask_hits, tr_chs, -1)
                            sig_strings = tr_chs // 36
                            sig_strings = [sig_strings[starts[i]:starts[i + 1]] for i in range(starts.shape[0] - 1)]
                            un_strings = [np.unique(s) for s in sig_strings]
                            num_un_strings = np.array([np.sum(s > 0) for s in un_strings])

                            ## make events mask to apply to NOT flat data
                            mask_strings = np.where(num_un_strings >= filters['strings'][0], True, False) \
                                           * np.where(num_un_strings <= filters['strings'][1], True, False)
                            Energy_prime = hf[f"{pk}/prime_prty/{part}/data"][:, 2]
                            Energy_mu = hf[f"{pk}/muons_prty/individ/{part}/data"][:, -1]
                            mask_energy = np.where(Energy_prime >= filters['E_prime'][0], True, False) \
                                          * np.where(Energy_prime <= filters['E_prime'][1], True, False) \
                                          * np.where(Energy_mu >= filters['E_indmu'][0], True, False) \
                                          * np.where(Energy_mu <= filters['E_indmu'][1], True, False)
                            ev_ids = hf[f"{pk}/ev_ids/{part}/data"][:]
                            if filters["filter_doubles"]:
                                _, un_idxs = np.unique(ev_ids, axis=0, return_index=True)
                                mask_double = np.zeros(ev_ids.shape[0], dtype=bool)
                                mask_double[un_idxs] = True
                                print(f"       Num of doubles deleted: {len(ev_ids) - sum(mask_double)}",
                                      file=open(path_to_log, 'a'))
                            else:
                                mask_double = True
                            # num of hits is to be included in the mask
                            mask_events_1 = mask_strings * mask_energy * mask_double

                            '''
                            Make new ev_starts. Since len of ev_starts may be less than initial len, new index in cicle is created.
                            At the same time make masks for hits and events based on filters.
                            '''
                            ## modify ev_starts
                            ev_starts = [0] * len(starts)
                            mask_events = np.copy(mask_events_1)
                            new_idx = 1
                            for j in range(1, len(starts)):
                                num_of_true = np.sum(mask_hits[starts[j - 1]:starts[j]])
                                # check num of hits and other filters
                                if filters['hits'][0] <= num_of_true < filters['hits'][1] and mask_events_1[j - 1]:
                                    ev_starts[new_idx] = ev_starts[new_idx - 1] + num_of_true
                                    new_idx += 1
                                else:
                                    mask_hits[starts[j - 1]:starts[j]] = False
                                    mask_events[j - 1] = False
                                    ev_starts.pop()
                            idxs_hits[i] = np.where(mask_hits == True)[0]
                            idxs_events[i] = np.where(mask_events == True)[0]
                            assert len(idxs_events[i]) == len(ev_starts) - 1

                            # save data
                            ev_starts_list[i] = ev_starts
                            num_un_strings_list[i] = num_un_strings[idxs_events[i]]
                            # get array of data in order to fast access
                            data = hf[f"{pk}/raw/data/{part}/data"][:]
                            data_to_copy[i] = data[idxs_hits[i]]
                            print(f"       After filter. events:{len(idxs_events[i])}, hits:{len(idxs_hits[i])}",
                                  file=open(path_to_log, 'a'))

                        elif origin == "ev_ids":
                            # get array of data in order to fast access
                            data = hf[f"{pk}/{origin}/{part}/data"][:]
                            data_to_copy[i] = data[idxs_events[i]]
                        elif origin == "prime_prty":
                            # get array of data in order to fast access
                            data = hf[f"{pk}/{origin}/{part}/data"][:]
                            data_to_copy[i] = data[idxs_events[i], 0:3]
                        elif origin == "muons_prty/individ":
                            # get array of data in order to fast access
                            data = hf[f"{pk}/{origin}/{part}/data"][:]         
                            data_to_copy[i] = data[idxs_events[i], -1:]
                        else:
                            pass
                        print(f"    Delta Time: {time.time() - time1:.2f}", file=open(path_to_log, 'a'))

                    data_to_copy = np.concatenate(data_to_copy, axis=0)

                    # flat data goes specially
                    if origin == "raw":
                        ev_starts_new = ev_starts_list.copy()
                        for i in range(1, len(ev_starts_list)):
                            ev_starts_new[i] = ev_starts_new[i - 1][-1] + np.array(ev_starts_list[i][1:])
                        ev_starts_new = np.concatenate(ev_starts_new, axis=0)
                        num_un_strings = np.concatenate(num_un_strings_list, axis=0)
                        hfout.create_dataset(f'{pk}/data', data=data_to_copy)
                        hfout.create_dataset(f'{pk}/ev_starts', data=ev_starts_new)
                        hfout.create_dataset(f'{pk}/num_un_strings', data=num_un_strings)
                    elif origin == "muons_prty/individ":
                        hfout.create_dataset(f'{pk}/{image}', data=data_to_copy)
                        assert all(data_to_copy > 0.)
                        hfout.create_dataset(f'{pk}/log10Emu', data=np.log10(data_to_copy))
                    else:
                        hfout.create_dataset(f'{pk}/{image}', data=data_to_copy)
    print(f"Totally passed = {time.time() - time0}")
    # write info about ouyput file
    collect_info(path_to_output + "/step1.h5", path_to_output, name="Step1FileStructure")
    return f"{h5_name}{postfix}"


if __name__ == "__main__":
    path_to_yml = 'steps_yml/step1.yml'
    config_dict = yaml.safe_load(Path(path_to_yml).read_text())
    step1(config_dict)
