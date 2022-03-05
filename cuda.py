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
    
def read_image(image_file, isGPU):
    with Image.open(image_file) as img:
        if(isGPU):
            return cp.array(img)
        else:
            return np.array(img)
    
def convert_image_to_single_channel(color_img, choice):
    if(choice == 'red'):
        return color_img[:,:,0]
    elif(choice == 'green'):
        return color_img[:,:,1]
    elif(choice == 'blue'):
        return color_img[:,:,2]

def plot_histogram(histogram, filename):
    idk, bins = np.histogram(histogram, bins=256, range=(0, 256))
    plt.title(filename)
    plt.figure()
    plt.plot(bins[0:-1], histogram)
    plt.savefig(safe_conf["OUTPUT_DIR"] + filename + ".png")
    plt.close()

def calc_histogram(img):
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
    
def kernel_median_filter():
    y = cp.RawKernel(r'''
                __global__ void denoise(uchar *d_grey, uchar *d_output, int matrixHeight, int matrixWidth, int numPixels)
                {
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    int row = blockIdx.y * blockDim.y + threadIdx.y;

                    unsigned char array[9];

                    if(col < matrixWidth && row < matrixHeight)
                    {
                        for(int x = 0; x < WINDOW_SIZE; x++)
                        {
                            for(int y = 0; y < WINDOW_SIZE; y++)
                            {
                                array[x*WINDOW_SIZE+y] = d_grey[(row+x-1)*matrixWidth+(col+y-1)];
                            }
                        }
                        for (int i = 0; i < 9; i++) {
                            for (int j = i + 1; j < 9; j++) {
                                if (array[i] > array[j]) { 
                                    //Swap the variables.
                                    unsigned char temp = array[i];
                                    array[i] = array[j];
                                    array[j] = temp;
                                }
                            }
                        }
                        d_output[rgb_ab] = (unsigned char) array[4];
                    }
                }
                     ''')

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
        
    # GPU only works on my PC/desktop
    # if(platform.system() != "Darwin"):
        # import GPUtil to check for GPU
    isGPU = GPUtil.getAvailable()
    print("GPU Available: " + isGPU)
        
    color_image = read_image(files[0], isGPU)
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])

    histogram = calc_histogram(img)
    plot_histogram(histogram, filenames[i])

if __name__ == "__main__":
    main()