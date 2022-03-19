import os
import platform
import sys
import time
import numpy as np
from numpy.random import default_rng
import matplotlib
import yaml
import random
from main import (
    plot_histogram,
    save_image,
    read_yaml,
    read_image,
    convert_image_to_single_channel
    )

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    Path(safe_conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    
    for i in range(len(files)):
        filenames[i] = os.path.basename(files[i])
        
    color_image = read_image(files[0])
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])

    # histogram = calc_histogram(img)
    # plot_histogram(histogram, filenames[i])

if __name__ == "__main__":
    main()