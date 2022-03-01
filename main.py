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
#     data = cp.array(image_file.shape[1]) # XXX
#     numPixels = image_file.shape[0] * image_file.shape[1]
#     histogram = cp.zeros(256)
#     # y(data, )

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
    # create numpy array of size 256 of zeroes
    histogram = np.zeros(256)
    
    row, col = img.shape
    
    for l in tqdm(range(int(256))):
        for i in range(int(row)):
            for j in range(int(col)):
                if(img[i][j] == l):
                    histogram[l] += 1
        print(histogram)

    return histogram

# Averaged histograms of pixel values for each class of images.  
def avg_hist():
    print("hi")
    
# Selected image quantization technique for user-specified levels
def image_quant(img):
    """Q = zeros(256,1); x = (0 : 255);
        >>fori = 1 : P
            Q = Q + r (i ) *((x >= t (i ))&(x <t (i + 1));
        end; % t (P + 1) = 256
        >>B = Q (A + 1);
    """
    q = np.zeros(256)
    print("bye")
    

# Salt and Pepper Method
# Mostly came from https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
# BUG why does multiplying with cons work, just strength should work i feel
def macaroni(img, strength):
    
    row, col = img.shape
    
    cons = 0.5
    
    ### add salt '255'
    # get num of pixels to salt
    num_pixels_white = random.randint(0, img.size)
    
    # multiply with strength
    num_pixels_white = strength * num_pixels_white * cons
    
    # create copy of image as to not modify original img
    img_copy = np.copy(img)
    
    # iterate through image's values and salt the values
    for i in range(int(num_pixels_white)):
                 
        # Pick a random x coordinate
        x=random.randint(0, col - 1)
                   
        # Pick a random y coordinate
        y=random.randint(0, row - 1)

        # Color that pixel to white
        img_copy[y][x] = 255
         
    ### add pepper '0'
    # get num of pixels to pepper
    num_pixels_black = random.randint(0, img.size)
    
    # multiply with user specified strength
    num_pixels_black = strength * num_pixels_black * (1 - cons)
    
    # iterate through image and add pepper
    for i in range(int(num_pixels_black)):
                 
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
# https://gist.github.com/Prasad9/28f6a2df8e8d463c6ddd040f4f6a028a
def domo_arrigato(img, strength):
    
    # set mean of guass distribution to 0
    mean = 0.0
        
    # using better statistical model
    rng = default_rng()
    # FIXME rng.normal(mean, strength, size=(row,col))
    noise = rng.normal(mean, strength, img.size)
    noise_reshape = noise.reshape(img.shape)

    copy_img = img + noise_reshape
    
    return copy_img

# linear filter
def linear_filter():
    print("linear filter")

# median filter
def median_filter():
    print("median filter")

# calculate mean square error
def mse(og_img, quantized_img):
    mserror = np.square(np.subtract(og_img, quantized_img)).mean
    
    return mserror
    

# [x] remember to make copies and work on those, DO NOT WORK ON OG IMAGES
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    # data_loc = Path(safe_conf['WIN_DATA_DIR'])
    
    # glob("*.bmp") uses regex which slows this down, nothing else is in the dir so dont need to use it
    # BUG files are not in order, files[0] == svar02.BMP
    files = list(data_loc.iterdir())
    # file_name = [i for i in range(len(files))]
    
    # create the output dir where all of the modified images will go
    Path(safe_conf['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    # Path(safe_conf["WIN_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    # for i in range(len(files)):
    #     file_name[i] = os.path.basename(files[i])
    
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
    # else:
    #     ts = time.perf_counter()
    #     # hist = calc_histogram(img)
    #     te = time.perf_counter()
    
    ts = time.perf_counter()
    hist = calc_histogram(img)
    # print(hist)
    # Image.fromarray(img).save("datasets/output/original.jpg")
    te = time.perf_counter()
    
    timings = []
    timings.append(te - ts)
    
    plt.hist(hist, bins=256, range=(0,256))
    plt.title(files[0])
    plt.savefig(safe_conf["OUTPUT_DIR"] + "blah.png")
    # plt.savefig(safe_conf["WIN_OUTPUT_DIR"] + "svar02.png")
    plt.close()
    
    # add salt & pepper noise to images
    # snp_img = macaroni(img, safe_conf["SNP_NOISE"])

    # add guassian noise to images
    # gaussian_img = domo_arrigato(img, safe_conf["G_NOISE"])
    
    # checking if images work !
    # FIXME noise works, files is out of order
    
    # salt = Image.fromarray(snp_img)
    # OSError: cannot write mode F as JPEG
    # https://stackoverflow.com/questions/21669657/getting-cannot-write-mode-p-as-jpeg-while-operating-on-jpg-image
    # should convert be 'RGB' or 'L'
    # gauss = Image.fromarray(gaussian_img).convert('L')
    
    # salt.save("datasets/output/salt.jpg", format="JPEG")
    # gauss.save("datasets/output/gauss.jpg", format="JPEG")



# [x]:
# work in batches, perf. timings, hist equalization, image quantization
# linear filter, median filter, averaged hist of pixel values for each class of images
# CUPY/CUDA
# optional: make GPU code OS agnostic
if __name__ == "__main__":
    main()