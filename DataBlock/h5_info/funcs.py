import os
import sys
from io import StringIO
from show_h5 import show_h5
import h5py as h5

def get_structure_h5(path):
    stdout = sys.stdout
    sys.stdout = StringIO()
    show_h5(path, show_attrs=True, show_data='none')
    x = sys.stdout.getvalue()
    sys.stdout = stdout
    return x

def get_h5_name(path_to_h5):
    # collect name
    for i, c in enumerate(path_to_h5[::-1]):
        if c == '/':
            break
    h5_name = path_to_h5[-i:-3]
    return h5_name

def collect_info(path_to_h5, path_to_output, name="FileStructure"):
    path_to_h5 = str(path_to_h5)
    path_to_output = str(path_to_output)
    path_to_info_file = f"{path_to_output}/{name}.txt"
    size = os.stat(path_to_h5)[6]
    print(f"Size is {size/2**30:.2f} GB", file=open(path_to_info_file,'w'))
    with h5.File(path_to_h5, 'r') as hf:
        keys1 = list(hf.keys())
        print(f"Keys1: {keys1}", file=open(path_to_info_file,'a'))
        keys2 = list(hf[keys1[0]].keys())
        print(f"Keys2: {keys2}", file=open(path_to_info_file,'a'))
    x = get_structure_h5(path_to_h5)
    print(x, file=open(path_to_info_file,'a'))