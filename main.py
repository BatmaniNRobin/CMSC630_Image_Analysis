import os
import platform
import sys
import time
import numpy as np
import matplotlib
import yaml
# import pandas as pd
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
        # if(isGPU):
        #     return cp.array(img)
        # else:
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
    elif(choice == 'blue'):
        return color_img[:,:,2]

# def kernel_calc_histogram(image_file, L):
#     import cupy as cp
#     y = cp.RawKernel(r'''
#                 __global__ void calcHistogram(unsigned char *data, int width, int numPixels, int *histogram) {
#                     int tid = blockDim.x * blockIdx.x + threadIdx.x;
#                     int row = tid / width;
#                     int column = tid - ((tid / width) * width);
#                     if (tid < numPixels) {
#                         int val = data[row * width + column];
#                         if (val != 0)
#                             atomicAdd(&histogram[val], 1);
#                     }
#                     return;
#                 }           
#             ''', 'calc_hist')
#     data = cp.array(image_file.shape[1]) # FIXME
#     numPixels = image_file.shape[0] * image_file.shape[1]
#     histogram = cp.zeros(256)
#     # y(data, )

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
    
# TODO translate my denoise code to work here if time
# def denoise():
#     cp.RawKernal = r'''__global__ void denoise(uchar *d_grey, uchar *d_output, int matrixHeight, int matrixWidth, int numPixels)
#         {
#             int col = blockIdx.x * blockDim.x + threadIdx.x;
#             int row = blockIdx.y * blockDim.y + threadIdx.y;

#             unsigned char array[9];

#             if(col < matrixWidth && row < matrixHeight)
#             {
#                 for(int x = 0; x < WINDOW_SIZE; x++)
#                 {
#                     for(int y = 0; y < WINDOW_SIZE; y++)
#                     {
#                         array[x*WINDOW_SIZE+y] = d_grey[(row+x-1)*matrixWidth+(col+y-1)];
#                     }
#                 }
#                 // // write value to d_output
#                 // d_output[rgb_ab] = (unsigned char) array[4];

#                 // bubblesort works ...
#                 for (int i = 0; i < 9; i++) {
#                     for (int j = i + 1; j < 9; j++) {
#                         if (array[i] > array[j]) { 
#                             //Swap the variables.
#                             unsigned char temp = array[i];
#                             array[i] = array[j];
#                             array[j] = temp;
#                         }
#                     }
#                 }
#                 d_output[rgb_ab] = (unsigned char) array[4];
#             }
#         }'''

# Salt and Pepper Method
# Mostly came from https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
# TODO add user specified strength
def macaroni(img, strength):
    
    row, col = img.shape
    
    # add salt '255'
    num_pixels_white = random.randint(0, img.size)
    
    img_copy = np.copy(img)
    
    for i in range(num_pixels_white):
                 
        # Pick a random x coordinate
        x=random.randint(0, col - 1)
                   
        # Pick a random y coordinate
        y=random.randint(0, row - 1)

        # Color that pixel to white
        img_copy[y][x] = 255
         
    # add pepper '0'
    number_of_pixels_black = random.randint(0, img.size)
    
    for i in range(number_of_pixels_black):
                 
        # Pick a random x coordinate
        x=random.randint(0, col - 1)
                   
        # Pick a random y coordinate
        y=random.randint(0, row - 1)

        # Color that pixel to black
        img_copy[y][x] = 0
         
    return img_copy

# adds guassaian noise to image
# utilzed similar methods as scikit, uses np.random_normal then adds noise back to image
# https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def domo_arrigato(img, sigma):
    print("domo_arrigato")
    

# TODO remember to make copies and work on those, DO NOT WORK ON OG IMAGES
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    # data_loc = Path(safe_conf['WIN_DATA_DIR'])
    
    # glob("*.bmp") uses regex which slows this down, nothing else is in the dir so dont need to use it
    files = list(data_loc.iterdir())
    
    # create the output dir where all of the modified images will go
    Path(safe_conf['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    # Path(safe_conf["WIN_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    
    # reads image into np array
    # TODO make this operate in batch/iterate through files
    color_image = read_image(files[0])
    
    # convert to greyscale / proper color channel
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])

    # these only work on my PC/desktop
    if(platform.system() != "Darwin"):
        # import GPUtil to check for GPU
        import GPUtil
        
        isGPU = GPUtil.getAvailable()
        print("GPU Available: ")
        print(isGPU)
        # hist = kernel_calc_histogram(img, len(img))
    else:
        ts = time.perf_counter()
        hist = calc_histogram(img)
        te = time.perf_counter()
    
    ts = time.perf_counter()
    # hist = calc_histogram(img)
    te = time.perf_counter()
    
    timings = []
    timings.append(te - ts)
    
    plt.hist(hist, bins=256, range=(0,256))
    plt.title("cyl01.BMP")
    plt.savefig(safe_conf["OUTPUT_DIR"] + "cyl01.png")
    # plt.savefig(safe_conf["WIN_OUTPUT_DIR"] + "cyl01.png")
    plt.close()
    
    # add salt & pepper noise to images
    snp_img = macaroni(img, safe_conf["SNP_NOISE"])

    # add guassian noise to images
    gaussian_img = domo_arrigato(img, safe_conf["G_NOISE"])
    
    # checking if images work !
    salt = Image.fromarray(snp_img)
    gauss = Image.fromarray(gaussian_img)
    salt.save("first.jpg", format="JPEG")
    gauss.save(gauss, format="JPEG")
    
    

if __name__ == "__main__":
    main()