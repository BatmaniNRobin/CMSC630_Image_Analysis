import os
import platform
import sys
import time
import numpy as np
from numpy.random import default_rng
import matplotlib
import yaml
import random

# GPU imports
import cupy as cp
import GPUtil

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
    
def calc_histogram(img):
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

    histogram = np.zeros(256)
    img_size = len(img)

    for l in tqdm(range(256)):
      for i in range(img_size):
        if img.flat[i] == l:
            histogram[l] += 1
            
    return histogram

def kernel_calc_histogram(image_file, L):
    y = cp.RawKernel(r'''
                __global__ void calcHistogram(unsigned char *data, int width, int numPixels, int *histogram) {
                    int tid = blockDim.x * blockIdx.x + threadIdx.x;
                    int row = tid / width;
                    int column = tid - ((tid / width) * width);
                    if (tid < numPixels) {
                        int val = data[row * width + column];
                        if (val != 0)
                            atomicAdd(&histogram[val], 1);
                    }
                    return;
                }           
            ''', 'calc_hist')
    data = cp.array(image_file.shape[1]) # XXX
    numPixels = image_file.shape[0] * image_file.shape[1]
    histogram = cp.zeros(256)
    # y(data, )

def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['WIN_DATA_DIR'])
    Path(safe_conf["WIN_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    
    for i in range(len(files)):
        filenames[i] = os.path.basename(files[i])
        
    color_image = read_image(files[0])
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
    
    # GPU only works on my PC/desktop
    # if(platform.system() != "Darwin"):
        # import GPUtil to check for GPU
    
    isGPU = GPUtil.getAvailable()
    print("GPU Available: " + isGPU)

    histogram = calc_histogram(img)
    
    idk, bin_edges = np.histogram(histogram, bins=256, range=(0,256))
    plt.figure()
    plt.title("Histogram")
    plt.plot(bin_edges[0:-1], histogram)
    plt.savefig(safe_conf["WIN_OUTPUT_DIR"] + "test.png")
    plt.close()

if __name__ == "__main__":
    main()