import os
import sys
import time
from turtle import color
import numpy as np
import matplotlib
import yaml
# TODO would the stricter typing of toml be easier?
# import toml 
# import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# TODO these would only work if on my PC/desktop
# import pycuda
# import cupy as cp
    
def read_yaml(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    
def read_image(image_file):
    with Image.open(image_file) as img:
        return np.array(img)

def convert_image_to_single_channel(color_img, choice):
    match choice:
        case 'red':
            return color_img[:,:,0]
        case 'green':
            return color_img[:,:,1]
        case 'blue':
            return color_img[:,:,2]

# TODO
def calc_histogram(image_file):
    """
        >>h=zeros(256,1);       OR           >>h=zeros(256,1);
        >>for l = 0 : 255                       >>for l = 0 : 255
            for i = 1 : N                           h(l +1)=sum(sum(A == l ));
                for j = 1 : M                       end
                    if (A(i ,j ) == l )             >>bar(0:255,h);
                        h(l +1) = h(l +1)+1;
                    end
                end
            end
        end
        
        >>bar(0:255,h);
    """
    # create numpy array of size 256 of zeroes
    h = np.zeros(256)
    # h = cp.zeros_pinned(256) # cupy
    
    N = 1 # TODO matrix size length
    M = 1 # TODO matrix size width
    
    for l in tqdm(range(int(255))):
        for i in range(int(N)):
            for j in range(int(M)):
                count = 0
    

# TODO remember to make copies and work on those, DO NOT WORK ON OG IMAGES
def main():
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    # data_loc = safe_conf['WIN_DATA_DIR']
    
    # glob("*.bmp") uses regex which slows this down, nothing else is in the dir so dont need to use it
    files = list(data_loc.iterdir())
    
    # FIXME creates the output dir where all of the modified images will go
    # Path.mkdir(safe_conf["OUTPUT_DIR"], exist_ok=True)
    
    # reads image into np array
    # TODO make this operate in batch/iterate through files
    color_image = read_image(files[0])
    
    # convert to greyscale / proper color channel
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
    # print(img)
    
    blah = calc_histogram(img)
    


if __name__ == "__main__":
    main()