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
# XXX would the stricter typing of toml be easier?

# XXX these would only work if on desktop
# import pycuda
# import cupy
import pathlib as Path
    
def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    
# def read_image(image_file):
#     return Image.open(image_file)

def main():
    global config_file
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    print(safe_conf)
    
    print(config_file[0])
    blah = config_file[0]
    # print(blah["DATA_DIR"])
    
    # p = Path(config_file["DIRECTORIES"])
    # for f in p.iterdir():
        # print(f)
    
    # img = read_image()
    # print(img)
    
if __name__ == "__main__":
    main()