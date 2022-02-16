import numpy as np
# import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
import time
from PIL import Image
import yaml

import pycuda
import cupy

def main(config_file):
    print("hello world")
    conf = yaml.safe_load(config_file)
    img = Image.open()
    print(img)
    
if __name__ == "__main__":
    main()