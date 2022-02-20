import os
import platform
import sys
import time
import numpy as np
import matplotlib
import yaml
# import pandas as pd

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
    # this only works for python 3.10
    '''match choice:
        case 'red':
            return color_img[:,:,0]
        case 'green':
            return color_img[:,:,1]
        case 'blue':
            return color_img[:,:,2]'''

    if(choice == 'red'):
        return color_img[:,:,0]
    elif(choice == 'green'):
        return color_img[:,:,1]
    else:
        return color_img[:,:,2]

def kernel_calc_histogram(image_file, L):
    import cupy as cp
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
    data = cp.array(image_file.shape[1]) # FIXME
    numPixels = image_file.shape[0] * image_file.shape[1]
    histogram = cp.zeros_pinned(256)
    # y(data, )

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
    histogram = np.zeros(256)
    
    N = image_file.shape[0] # matrix size height
    M = image_file.shape[1] # matrix size width
    
    for l in tqdm(range(int(256))):
        for i in range(int(N)):
            for j in range(int(M)):
                if(image_file[i][j] == l):
                    histogram[l] += 1

    return histogram
    

# TODO remember to make copies and work on those, DO NOT WORK ON OG IMAGES
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    # data_loc = Path(safe_conf['DATA_DIR'])
    data_loc = Path(safe_conf['WIN_DATA_DIR'])
    
    # glob("*.bmp") uses regex which slows this down, nothing else is in the dir so dont need to use it
    files = list(data_loc.iterdir())
    
    # FIXME creates the output dir where all of the modified images will go
    Path(safe_conf["WIN_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    # reads image into np array
    # TODO make this operate in batch/iterate through files
    color_image = read_image(files[0])
    
    # convert to greyscale / proper color channel
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
    
    if(platform.system() != "Darwin"):
        # TODO these would only work if on my PC/desktop
        # import GPUtil to check for GPU
        import GPUtil
        
        isGPU = GPUtil.getAvailable()
        print("GPU Available: ")
        print(isGPU)
        # hist = kernel_calc_histogram(img, len(img))
    else:
        hist = calc_histogram(img)
    
    hist = calc_histogram(img)
    
    plt.hist(img, bins=256, range=(0,256))
    plt.title("cyl01.BMP")
    plt.savefig("cyl01.png")
    plt.close()
    

if __name__ == "__main__":
    main()