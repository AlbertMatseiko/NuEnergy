import yaml
from pathlib import Path

from step1 import step1
from step2 import step2
from step3 import step3
from DataAnalysis.make_my_plots import make_distr_plots
from DataAnalysis.weights import calc_weights_distr

if __name__ == '__main__':
    '''
    Filtering pure MC. Config is to be written at step1.yml
    Works for about 10 mins
    See logs and output file structure in:
        ./data/<name_of_file>/step1_logs.txt
        ./data/<name_of_file>/Step1FileStructure.txt
    '''
    path_to_yml_1 = Path(__file__).parent.absolute() / 'steps_yml' / 'step1.yml'
    config_dict_1 = yaml.safe_load(Path(path_to_yml_1).read_text())
    h5_dir_name = step1(config_dict_1) #"baikal_multi_1223_flat_signal_H5_S2_Emu10to1000000"#
    print("First step is done.")

    '''
    Splitting to train-test-val. Config is to be written at step2.yml
    See logs and output file structure in:
        ./data/<name_of_file>/step2_<nu_type>_logs.txt
        ./data/<name_of_file>/Step2_<nu_type>FileStructure.txt
    Works for about 5 mins
    '''
    path_to_yml_2 = Path(__file__).parent.absolute() / 'steps_yml' / 'step2.yml'
    config_dict_2 = yaml.safe_load(Path(path_to_yml_2).read_text())
    config_dict_2['filtered_h5_name'] = h5_dir_name
    step2_name = step2(config_dict_2)  #, filtered_h5_name=h5_dir_name)
    print("Second step is done.")

    '''
       Normilizing data. Config is to be written at step3.yml
       See logs and output file structure in: 
           ./data/<name_of_file>/step3_<nu_type>_logs.txt
           ./data/<name_of_file>/Step2_<nu_type>FileStructure.txt
       Works for about 1 min
       '''
    path_to_yml_3 = Path(__file__).parent.absolute() / 'steps_yml' / 'step3.yml'
    config_dict_3 = yaml.safe_load(Path(path_to_yml_3).read_text())
    config_dict_3['h5_name'] = h5_dir_name
    config_dict_3['step_name'] = step2_name
    step3_name = step3(config_dict_3)
    print("Third step is done.")

    h5_step, list_of_keys, Q_max = f"{step3_name}.h5", ['train', 'test', 'val'], 0.2
    path = Path(__file__).parent / f"data/{config_dict_3['h5_name']}/{step3_name}.h5"
    _ = calc_weights_distr(str(path))
    make_distr_plots({config_dict_3['h5_name']}, h5_step, list_of_keys, Q_max)
    
