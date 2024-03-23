# Info about dataset creating procedures

## FYI
All datasets are created from file *baikal_multi_1223_flat.h5*. That contains raw Monte-Carlo simulation of events (both track and cascades) within 1 cluster of BaikalGVD from 3 sources: 
- Atmospheric muons, *muatm* key
- Atmospheric neutrino, *nuatm* key
- Astrophysical neutrino, *nu2* key: stands for simulations of neutrino with e-2 high energy spectrum.

Within the task of netrino energy reconstruction only *nuatm* and *nu2* are used.

Events in *baikal_multi_1223_flat.h5* within one source (particle) key are grouped into *parts* that correspond to origin root MC files.

Each of event in *baikal_multi_1223_flat.h5* consists of arbitary number of hits. Therefore, in order of memory efficiency, info about distinct hits of all the events is located in a single array *data* (e.g. "nuatm/raw/data/part_1001/data"), and starts of corresponding events are loacated as indexes of the array in *ev_starts* dir (e.g. "nuatm/raw/ev_starts/part_1001/data")

## Step by step procedure
Description of how ```make_h5_from_PureMCh5.py``` script works.
Configurations of output dataset are to be written in ```steps_yml/step_{number}.yml```.

### Step 1: concatenates parts and applies filters to hits and events.
Reads configuration from ```steps_yml/step_1.yml```, calls function *step1* from ```step1.py``` script. That in term iterates over *particle key* and *parts* in *baikal_multi_1223_flat.h5*, collects data from different *parts* into one list and, simuntaniously, generates masks to apply on hits-array and, then, on all the events.

Masks are generated in according to configuration, set in step_1.yml:
- ```path_to_h5``` $-$ absolute (!) path to input file with data
- ```bad_parts``` $-$ list of parts which are originated from root MC files with bugs. To be excluded.
- ```pk_list``` $-$ list of particle keys to be included in output dataset
- ```fields_to_map``` $-$ correspondance of overcomplicated keys in *baikal_multi_1223_flat.h5* to simple keys in output dataset
- ```filters``` $-$ filters to apply on values in data
    - ```only_signal``` $-$ if to take only hits caues by particles and not the noise sources
    - ```filter_doubles```. Since there are events that contains 2 (or more) interacting muons, they were doubled in *baikal_multi_1223_flat.h5* in order to contain information about that muons separately. Set True to remove events with more than 1 muon
    - ```Q``` $-$ limits on charge detected by hit
    - ```hits``` $-$ limits on number of hits per event in output dataset
    - ```strings``` $-$ limits on number of triggered strings within cluster per event in output dataset
    - ```E_prime``` $-$ limits on energy of primapy neutrino
    - ```E_indmu``` $-$ limits on energy of individual muon recalculated in the center-plane of cluster

*step1* crates dataset directory (e.g. "data/baikal_multi_1223_flat_signal_H5_S2") with name, generated according to configuration. Then, inside, it creates the output *HDF5* dataset (e.g. "data/baikal_multi_1223_flat_signal_H5_S2/step1.h5") and saves information about origin file (e.g. "data/baikal_multi_1223_flat_signal_H5_S2/OriginFileStructure.txt") and logs (e.g. "data/baikal_multi_1223_flat_signal_H5_S2/step1_logs.txt").

As function, *step1* returns the path to created dataset, that in term goes as input for *step2*.

### Step 2: separate data into train/test/val datasets.

Function *step2* from ```step2.py``` script collects data from step1.h5 and devides it into train/test/val datasets according to ```steps_yml/step_2.yml``` configurations. That are:

- ```filtered_h5_name``` $-$ name of directory, where step1.h5 file was created. 
- ```num_in_train``` $-$ number of events to add into 'train' dataset
- ```num_in_test``` $-$ number of events to add into 'test' dataset

'val' dataset contains evetns from *step1.h5* file, that were not added to 'train' or 'test'.

It also SELECTS EVENTS such that ```log10Emu``` energy spectra is uniform!

*step2* creates the output *HDF5* dataset (e.g. "baikal_multi_1223_flat_signal_H5_S2/step2_all_1_1_train600000_test40000.h5") and saves logs (e.g. "baikal_multi_1223_flat_signal_H5_S2/step2_all_1_1_train600000_test40000.txt").

As function, *step2* returns the path to created dataset, that in term goes as input for *step3*.

### Step 3: normalize data.

The most boring step. 

*step3* funciton from ```step3.py``` normalizes the hits data such that means are 0 and stds are 1 for each channel. Also norms energy if needed. Configure is to be written in ```steps_yml/step_3.yml```. 

Adds calculated means and stds to the output *HDF5* file (e.g. "baikal_multi_1223_flat_signal_H5_S2/step3_all_1_1_train200000_test10000_normlog10E_normed.h5"). Also saves logs in txt.

#### After 3rd step, the dataset step3_\<smth\>.h5 is ready to train NN on!

## Final
After that, some distribution plots for dataset are created. Then energy weights distribution is calculated and saved in ```weights_distr_train.npy``` in order to be able to flatten energy spectrum during training.