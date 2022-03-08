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
    
def plot_histogram(histogram, filename, applied_method):
    idk, bins = np.histogram(histogram, bins=256, range=(0, 256))
    plt.title("working?")
    plt.figure()
    plt.plot(bins[0:-1], histogram)
    plt.savefig(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".png")
    plt.close()
    
def save_image(img, filename, applied_method):
    # OSError: cannot write mode F as JPEG
    # https://stackoverflow.com/questions/21669657/getting-cannot-write-mode-p-as-jpeg-while-operating-on-jpg-image
    # should convert be 'RGB' or 'L'
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

# incase this breaks again:
## https://datacarpentry.org/image-processing/05-creating-histograms/
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
    # this works PHENOMONALLY better than 2 for loops
    histogram = np.zeros(256)
    img_size = len(img)

    for l in tqdm(range(256)):
      for i in range(img_size):
        if img.flat[i] == l:
            histogram[l] += 1

    return histogram

# create our cumulative sum function
# taken from: https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23
# I USED NP.CUMSUM() INSTEAD THOUGH
# def cumsum(a):
#     a = iter(a)
#     b = [next(a)]
#     for i in a:
#         b.append(b[-1] + i)
#     return np.array(b)

# https://medium.com/analytics-vidhya/image-equalization-contrast-enhancing-in-python-82600d3b371c
# https://medium.com/@kyawsawhtoon/a-tutorial-to-histogram-equalization-497600f270e2
# equalization
# Selected image quantization technique for user-specified levels
def equalization(histogram, img):
    '''Q = zeros(256,1); x = (0 : 255);
        >>fori = 1 : P
            Q = Q + r (i ) *((x >= t (i ))&(x <t (i + 1));
        end; % t (P + 1) = 256
        >>B = Q (A + 1);
    '''
    # flatten array
    img_flattened = img.flatten()

    # cumulative sum
    cum_sum = np.cumsum(histogram)
    
    ## normalize values between 0-256
    normalization = (cum_sum - cum_sum.min()) * 255
    n = cum_sum.max() - cum_sum.min()
    
    uniform_norm = normalization / n
    uni = uniform_norm.astype(np.uint8)
    
    # flattened histogram
    image_eq = uni[img_flattened]
    quantized = np.reshape(image_eq, img.shape)

    return calc_histogram(image_eq), quantized

# Salt and Pepper Method
# Mostly came from https://www.geeksforgeeks.org/add-a-salt-and-pepper-noise-to-an-image-with-python/
# BUG why does multiplying with cons work, just strength should work i feel
def salt_pepper(img, strength):
    
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
def gaussian(img, strength):
    
    # set mean of guass distribution to 0
    mean = 0.0

    row, col = img.shape
        
    # using better statistical model
    rng = default_rng()
    noise = rng.normal(mean, strength, size=(row,col))

    noise_reshape = noise.reshape(img.shape)

    copy_img = img + noise_reshape
    
    return copy_img

# linear filter
# TODO should smoothen image but idk why it does nothing
def linear_filter(img, weights):
      
    kernel = np.array(weights) # 3 x 3
  
    rows, cols = img.shape # 768 x 568
    mask_rows, mask_cols = kernel.shape # 3 x 3

    copy_img = np.zeros((rows, cols))

    # iterate through img
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
        # iterate through filter
            for i in range(mask_rows):
                for j in range(mask_cols):
                    intensity = img[row - i + 1, col - j + 1]
                    kernel_value = kernel[i, j]
                    copy_img[row, col] += (intensity * kernel_value)
    return copy_img

# median filter
# https://stackoverflow.com/questions/26870349/two-dimensional-array-median-filtering
# https://www.geeksforgeeks.org/noise-removal-using-median-filter-in-c/
# https://www.geeksforgeeks.org/spatial-filters-averaging-filter-and-median-filter-in-image-processing/
def median_filter(img, weights):
  
  kernel = np.array(weights) # array of weights [0,0,0,0,1,0,0,0,0]

  rows, cols = img.shape # 768 x 568
  kernel_rows, kernel_cols = kernel.shape # 3 x 3

  window = np.zeros(kernel.size) # [0,0,0,0,0,0,0,0,0]

  copy_img = np.zeros((rows, cols)) # 768 x 568

  # iterate through img
  for row in range(1, rows - 1):
    for col in range(1, cols - 1):
      
      pixel = 0
      for i in range(kernel_rows):
        for j in range(kernel_cols):
          # store neighbor pixel values in window
          window[pixel] = img[row - i + 1][col - j + 1]
          pixel += 1
      # TODO pad array with weights here

      window.sort()

      copy_img[row][col] = window[pixel // 2]
      
  copy_img = copy_img.astype(np.uint8)

  return copy_img
    
# calculate mean square error
# https://www.geeksforgeeks.org/python-mean-squared-error/
def mse(og_img, quantized_img):
    mserror = (np.square(np.subtract(og_img, quantized_img))).mean()
    
    return mserror
    
def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    # create the output dir where all of the modified images will go
    Path(safe_conf['OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
    
    # files are not in order, files[0] == svar02.BMP
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    r, c = 256, 7
    average_classes = [[0 for i in range(r)] for y in range(c)]
    num = [0 for i in range(c)]
    
    # TODO empty list to record length of time for each process?
    # should everything be appended to a list for evaluation metrics?
    timings = []
    
    for i in range(len(files)):
        filenames[i] = os.path.basename(files[i])
        
        if (".BMP" in filenames[i]):
            filenames[i] = os.path.splitext(filenames[i])[0]
    
        color_image = read_image(files[i])
        
        # convert to greyscale / proper color channel
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        
        # create histograms
        # ts = time.perf_counter()
        histogram = calc_histogram(img)
        plot_histogram(histogram, filenames[i], "_hist")
        # te = time.perf_counter()
        # timings.append(te - ts)
        
        # sum of each hist for each class
        if("cyl" in filenames[i]):
            average_classes[0] += histogram
            num[0]+= 1
            
        elif("inter" in filenames[i]):
            average_classes[1] += histogram
            num[1]+= 1
            
        elif("let" in filenames[i]):
            average_classes[2] += histogram
            num[2]+= 1
            
        elif("mod" in filenames[i]):
            average_classes[3] += histogram
            num[3]+= 1
            
        elif("para" in filenames[i]):
            average_classes[4] += histogram
            num[4]+= 1
            
        elif("super" in filenames[i]):
            average_classes[5] += histogram
            num[5]+= 1
            
        elif("svar" in filenames[i]):
            average_classes[6] += histogram
            num[6]+= 1
        
        # add salt & pepper noise to images then save
        snp_img = salt_pepper(img, safe_conf["SNP_NOISE"])
        salt = save_image(snp_img, filenames[i], "_salt")

        # add guassian noise to images then save
        gaussian_img = gaussian(img, safe_conf["G_NOISE"])
        gauss = save_image(gaussian_img, filenames[i], "_gauss")
        
        # create equalized histogram and quantized image
        equalized, quantized = equalization(histogram, img)
        plot_histogram(equalized, filenames[i], "_equalized")
        quant = save_image(quantized, filenames[i], "_quantized")
        
        # calculate mean square error
        msqe = mse(img, quantized)
        
        # apply linear filter to salted images
        linear = linear_filter(snp_img, safe_conf["LINEAR_WEIGHT"])
        not_salted = save_image(linear, filenames[i], "_linear")
        
        # apply median filter to salted images
        median = median_filter(snp_img, safe_conf["MEDIAN_WEIGHT"])
        unsalted = save_image(median, filenames[i], "_median")
    
    # Averaged histograms of pixel values for each class of images.
    for y in range(c):
        for x in range(r):
            average_classes[y][x] = int(average_classes[y][x] / num[y])
    
    plot_histogram(average_classes[0], "cyl", "_avg")
    plot_histogram(average_classes[1], "inter", "_avg")
    plot_histogram(average_classes[2], "let", "_avg")
    plot_histogram(average_classes[3], "mod", "_avg")
    plot_histogram(average_classes[4], "para", "_avg")
    plot_histogram(average_classes[5], "super", "_avg")
    plot_histogram(average_classes[6], "svar", "_avg")
        
        
# [x]: performance timings

# CUPY/CUDA
# optional: make GPU code OS agnostic, threading/speedup
if __name__ == "__main__":
    main()