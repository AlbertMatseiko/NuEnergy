import numpy as np
import h5py as h5
import time
import yaml
from pathlib import Path
from DataAnalysis.funcs import collect_info

# generate postfix to step2 h5 name from config
def generate_postfix(config_dict):
    num_train = config_dict['num_in_train']
    num_test = config_dict['num_in_test']
    return f"_FlatE_train{num_train}_test{num_test}"

# mask events to make flat energy spectra
def idxs_flatten_spec(E):
    start, stop, bins = np.log10(10), 6., 200
    e_range = np.linspace(start, stop, bins+1)

    nums = []
    for bin_start, bin_stop in zip(e_range[:-1], e_range[1:]):
        nums.append(E[(E>bin_start) * (E<=bin_stop)].shape[0])
    num_per_bin = min(nums)
        
    idxs = np.array([], dtype=int)
    for bin_start, bin_stop in zip(e_range[:-1], e_range[1:]):
        bin_mask = ((E>bin_start) * (E<=bin_stop)) #masks all events inside current bin
        idxs_bin = np.where(bin_mask)[0]
        idxs = np.append(idxs, idxs_bin[:num_per_bin], axis=0) #append (only a number) of global idxs of events inside current bin
    return np.random.permutation(idxs)

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

def get_new_flat_data(idxs, starts, data):
    new_starts = np.zeros(len(idxs)+1, dtype=int)
    for i_new, i_old in enumerate(idxs):
        new_starts[i_new+1] = new_starts[i_new] + (starts[i_old+1]-starts[i_old])
        
    lengs = np.diff(new_starts)
    new_data = np.zeros((sum(lengs),5), dtype=np.float32)
    for i_new, i_old in enumerate(idxs):
        new_data[new_starts[i_new]:new_starts[i_new+1]] = data[starts[i_old]:starts[i_old+1]]
    return new_starts, new_data

# MAIN
def step2(config_dict, filtered_h5_name=None):
    '''
    Routine to split filtered h5 dataset in train, test and val datasets.
    '''
    
    # declare paths to output and logs
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
    print(f"Start of step2!", file=open(path_to_log, 'w'))
    
    """start reading and writing data"""
    # concatenate particles
    with h5.File(path_to_filtered_h5, 'r') as hf:
        with h5.File(path_to_output_h5, 'w') as hfout:
            D = dict(data=[], ev_ids=[], ev_starts=[], 
                     individ_muon_energy=[], log10Emu=[], 
                     num_un_strings=[], prime_prty=[])
            for key, arr in D.items():
                for particle in hf.keys():
                    arr.append(hf[f'{particle}/{key}'][:])
                if key=='ev_starts':
                    arr[1] = (arr[0][-1]-arr[0][0]) + (arr[1][1:]-arr[1][0])
                arr = np.concatenate(arr, axis=0)
                
                hfout.create_dataset(f"all/{key}", data=arr)
                arr = []
    print(f"Dataset for 'all' is created", file=open(path_to_log, 'a'))
                
    # create train/test/val datasets       
    with h5.File(path_to_output_h5, 'a') as hf:
        E = hf['all/log10Emu'][:]
        idxs = idxs_flatten_spec(E)
        print(f"Idxes are loaded", file=open(path_to_log, 'a'))
        
        starts = hf['all/ev_starts'][:]
        data = hf['all/data'][:]
        starts, data = get_new_flat_data(idxs, starts, data)
        print(f"Flat data is loaded", file=open(path_to_log, 'a'))
        #print(len(data))
        #print(len(starts))
        #print(starts[0], starts[-1])
        assert len(starts)==len(idxs)+1
        assert len(data)==starts[-1]
        assert len(data)==np.diff(starts).sum()
        
        num_in_train = config_dict['num_in_train']
        num_in_test = config_dict['num_in_test']
        start = 0
        stop = num_in_train+1
        hf.create_dataset(f"train/ev_starts", data=starts[start:stop]-starts[start])
        hf.create_dataset(f"train/data", data=data[starts[start]:starts[stop]])
        print(f"Train flat data is uploaded", file=open(path_to_log, 'a'))
        
        start = num_in_train
        stop = num_in_test+num_in_train+1
        hf.create_dataset(f"test/ev_starts", data=starts[start:stop]-starts[start])
        hf.create_dataset(f"test/data", data=data[starts[start]:starts[stop]])
        print(f"Test flat data is uploaded", file=open(path_to_log, 'a'))
        
        start = num_in_train + num_in_test
        hf.create_dataset(f"val/ev_starts", data=starts[start:]-starts[start])
        hf.create_dataset(f"val/data", data=data[starts[start]:starts[-1]])
        print(f"Val flat data is uploaded", file=open(path_to_log, 'a'))
        
        starts, data = 0, 0
        print(f"Flat data is splited", file=open(path_to_log, 'a'))
        del hf[f'all/data']
        del hf[f'all/ev_starts']
        
    for key in ['ev_ids', 'individ_muon_energy', 'log10Emu', 'num_un_strings', 'prime_prty']:
        print(f'{key}')
        with h5.File(path_to_output_h5, 'r') as hf:
            train = hf[f'all/{key}'][:]
            train = train[idxs[:num_in_train]]
            
            test = hf[f'all/{key}'][:]
            test = test[idxs[num_in_train:num_in_train+num_in_test]]
            
            val = hf[f'all/{key}'][:]
            val = val[idxs[num_in_train+num_in_test:]][:]
            print(f"{key} is readed", file=open(path_to_log, 'a'))
        with h5.File(path_to_output_h5, 'a') as hf:
            hf.create_dataset(f"train/{key}", data=train)
            hf.create_dataset(f"test/{key}", data=test)
            hf.create_dataset(f"val/{key}", data=val)
            del hf[f'all/{key}']
            print(f"{key} is splited", file=open(path_to_log, 'a'))
            train, test, val = 0, 0, 0
    print(f"All data is splited!", file=open(path_to_log, 'a'))
        
    collect_info(path_to_output_h5, path_to_output, name=f"Step2{postfix}FileStructure")
    print(f"Totally passed = {time.time() - time0}", file=open(path_to_log, 'a'))      
    return f"step2{postfix}"

if __name__ == "__main__":
    path_to_yml = Path(__file__).parent.absolute() / 'steps_yml/step2v2.yml'
    config_dict = yaml.safe_load(Path(path_to_yml).read_text())
    step2(config_dict)