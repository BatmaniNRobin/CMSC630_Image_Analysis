from tokenize import String
import numpy as np
# import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
import time
from PIL import Image
import yaml
# import toml 
# TODO would the stricter typing of toml be easier?

# TODO these would only work if on desktop
# import pycuda
# import cupy
import pathlib as Path
    
def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    
def read_image(image_file):
    return Image.open(image_file)

def get_file_names(data_loc):
    file_names = []
    p = Path.Path(data_loc)
    
    # TODO make this actually iterate
    for f in p.iterdir():
        file_names[f] = f
    return file_names
    
    # TODO this might work too, worth looking into
    # tmp = Path('tmp')
    # dgen = tmp.iterdir()
    # list(dgen)

def main():
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = safe_conf['DIRECTORIES']['DATA_DIR']
    win_data_loc = safe_conf['DIRECTORIES']['WIN_DATA_DIR']
    
    file_names = get_file_names(win_data_loc)
    


if __name__ == "__main__":
    main()