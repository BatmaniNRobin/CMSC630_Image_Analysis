import os
import platform
import sys
import time
import numpy as np
from numpy.random import default_rng
import matplotlib
import yaml
import random

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    
def read_image(image_file):
    with Image.open(image_file) as img:
        return np.array(img)
    
def convert_image_to_single_channel(color_img, choice):
    if(choice == 'red'):
        return color_img[:,:,0]
    elif(choice == 'green'):
        return color_img[:,:,1]
    elif(choice == 'blue'):
        return color_img[:,:,2]

# TODO holy shit whats wrong with my histograms
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    files = list(data_loc.iterdir())
    
    Path(safe_conf['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    color_image = read_image(files[0])
    
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])

    histogram = np.zeros(256)
    
    # row, col = img.shape
    
    # for l in tqdm(range(int(256))):
    #     for i in range(int(row)):
    #         for j in range(int(col)):
    #             if(img[i][j] == l):
    #                 histogram[l] += 1
    # Get size of pixel array
    N = len(img)

    for l in range(256):
      for i in range(N):
        if img.flat[i] == l:
            histogram[l] += 1
    
    idk, bin_edges = np.histogram(histogram, bins=256, range=(0,256))
    plt.figure()
    plt.title("Histogram")
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig(safe_conf["OUTPUT_DIR"] + "test.png")
    plt.close()

if __name__ == "__main__":
    main()