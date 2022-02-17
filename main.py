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
# import toml would the stricter typing of toml be easier?

# import pycuda
# import cupy
    
def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def main():
    print("hello world")
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    img = Image.open(dataset/)
    print(img)
    
if __name__ == "__main__":
    main()