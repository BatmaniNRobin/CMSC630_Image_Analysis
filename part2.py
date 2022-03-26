import os
import platform
import sys
import time
from warnings import filters
import numpy as np
from numpy.random import default_rng
import matplotlib
import yaml
import random
from main import (
    plot_histogram,
    read_yaml,
    read_image,
    convert_image_to_single_channel
    )

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

# edge detection algos

# https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e
## sobel
def sobel_or_prewitt(img, edgeMethod):
    M, N = img.shape
    
    if(edgeMethod == "prewitt"):
        Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)
        Ky = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], np.float32)
    else:
        Kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], np.float32)
    
    copy_img = np.zeros((M, N))
    
    for i in tqdm(range(1, M - 1)):
        for j in range(1, N - 1):
            hFilter = (Kx[0, 0] * img[i - 1, j - 1]) + \
                        (Kx[0, 1] * img[i - 1, j]) + \
                        (Kx[0, 2] * img[i - 1, j + 1]) + \
                        (Kx[1, 0] * img[i, j - 1]) + \
                        (Kx[1, 1] * img[i, j]) + \
                        (Kx[1, 2] * img[i, j + 1]) + \
                        (Kx[2, 0] * img[i + 1, j - 1]) + \
                        (Kx[2, 1] * img[i + 1, j]) + \
                        (Kx[2, 2] * img[i + 1, j + 1])

            vFilter = (Ky[0, 0] * img[i - 1, j - 1]) + \
                        (Ky[0, 1] * img[i - 1, j]) + \
                        (Ky[0, 2] * img[i - 1, j + 1]) + \
                        (Ky[1, 0] * img[i, j - 1]) + \
                        (Ky[1, 1] * img[i, j]) + \
                        (Ky[1, 2] * img[i, j + 1]) + \
                        (Ky[2, 0] * img[i + 1, j - 1]) + \
                        (Ky[2, 1] * img[i + 1, j]) + \
                        (Ky[2, 2] * img[i + 1, j + 1])
                           
            edgeMag = np.sqrt(pow(hFilter, 2.0) + pow(vFilter, 2.0))
            copy_img[i - 1, j - 1] = edgeMag
            
    return copy_img
    
    
# https://betterdatascience.com/implement-convolutions-from-scratch-in-python/
## canny (bonus point for others)
##### noise reduction
##### gradient calc
##### non-max suppression
##### double threshold
##### hysteresis - edge tracking

### applies filter to img
def convolve(img, filter):
    M, N = img.shape
    kernel_row, kernel_col = filter.shape
    
    copy_img = np.zeros((M - kernel_row + 1, N - kernel_col + 1))
    
    for row in tqdm(range(M - kernel_row + 1)):
        for col in range(N - kernel_col + 1):
            for i in range(kernel_row):
                for j in range(kernel_col):
                    value = img[row + i, col + j]
                    weight = filter[i, j]
                    copy_img[row, col] += value * weight
                    
    return copy_img

### applies 5x5 gaussian blur for canny - smoothens img to reduce noise
def gaussian_kernel(size):
    sigma = 1
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss_img = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    
    return gauss_img

### applies sobel filter for gradient
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], np.float32)
    
    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return G, theta

### reduces thick edges
def non_max_suppression(img, theta):
    M,N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                
                ### angle 0
                if (0<= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]


                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
                
            except IndexError as e:
                pass
    return Z

### identifies strong, weak, and non-relevant pixels of edges
### BUG guide used weak as 75, and strongThreshold as 0.15
def threshold(img, weak_pixel=25, strong_pixel=255, weakThreshold=0.05, strongThreshold=0.09):
    highThreshold = img.max() * strongThreshold;
    lowThreshold = highThreshold * weakThreshold;

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

### finalizes edges and transforms weak pixels to strong so edge is consistent
def hysteresis(img, weak, strong):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img

    
# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py
# TODO make sure you know how this all works
def canny_edge_detector(img):
    gauss_img = gaussian_kernel(5) # 5x5 window
    
    blurred_img = convolve(img, gauss_img)
    
    sobel, theta = sobel_filters(blurred_img)
    
    suppression = non_max_suppression(sobel, theta)
    
    threshold_img, weak, strong = threshold(suppression)
    
    canny_img = hysteresis(threshold_img, weak, strong)
    
    return canny_img

# erosion
def binarize(img, threshold_value=127):
    color_1 = 255
    color_2 = 0
    initial_conv = np.where((img <= threshold_value), img, color_1)
    final_conv = np.where((initial_conv > threshold_value), initial_conv, color_2)
    
    return final_conv

# https://python.plainenglish.io/image-erosion-explained-in-depth-using-numpy-320c01b674a8

# def erosion(img, erosion_level=3):
#     erosion_level = 3 if erosion_level < 3 else erosion_level

#     structuring_kernel = np.full(shape=(erosion_level, erosion_level), fill_value=255)
#     binary_img = binarize(img)
    
#     img_row, img_col = binary_img.shape
#     pad_width = erosion_level - 2
    
#     # pad img
#     img_pad = np.pad(binary_img, pad_width,'constant')
#     pad_row, pad_col = img_pad.shape
    
#     h_reduce, w_reduce = (pad_row - img_row), (pad_col - img_col)

#     # sub matrices of kernel size
#     flat_submatrices = np.array([
#         img_pad[i:(i + erosion_level), j:(j + erosion_level)]
#         for i in range(pad_row - h_reduce) for j in range(pad_col - w_reduce)
#     ])

#     # condition to replace the values - if the kernel equal to submatrix then 255 else 0
#     image_erode = np.array([255 if (i == structuring_kernel).all() else 0 for i in flat_submatrices])
#     image_erode = image_erode.reshape(binary_img.shape)

    
#     return image_erode


# https://medium.com/@ami25480/morphological-image-processing-operations-dilation-erosion-opening-and-closing-with-and-without-c95475468fca
def erosion(binary_img):
    M,N = binary_img.shape
    
    SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
    constant = 1
    
    copy_img = np.zeros((M,N), dtype=np.uint8)
    
    for i in range(constant, M-constant):
      for j in range(constant,N-constant):
            temp= binary_img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*SE
            copy_img[i,j]= np.max(product)
    
    return copy_img

# https://python.plainenglish.io/image-dilation-explained-easily-e085c47fbac2
# dilation
def dilation(binary_img):
    M,N = binary_img.shape
    
    k = 3 # defines window
    SE = np.ones((k,k), dtype=np.uint8)
    constant = (k-1) // 2
    
    copy_img = np.zeros((M,N), dtype=np.uint8)
    
    for i in range(constant, M-constant):
      for j in range(constant,N-constant):
            temp= binary_img[i-constant:i+constant+1, j-constant:j+constant+1]
            product= temp*SE
            copy_img[i,j]= np.min(product)
    
    return copy_img



# segmentation techniques
# hist thresholding
# clustering - k-means (bonus points for other ones)


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
        
        if (".BMP" in filenames[i]):
            filenames[i] = os.path.splitext(filenames[i])[0]
        
        color_image = read_image(files[i])
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        
        sobel = sobel_or_prewitt(img, "sobel")
        save_image(sobel, filenames[i], "_sobel")
        
        prewitt = sobel_or_prewitt(img, "prewitt")
        save_image(prewitt, filenames[i], "_prewitt")
        # following this: http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
        # https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
        # results in a worse/darker version of prewitt/sobel using my convolve and is slower but less code

        canny = canny_edge_detector(img)
        save_image(canny, filenames[i], "_canny")
        
        binary_img = binarize(img)
        save_image(binary_img, filenames[i], "_binary")
        
        eroded = erosion(binary_img)
        save_image(eroded, filenames[i], "_erosion")
        
        dilated = dilation(binary_img)
        save_image(dilated, filenames[i], "_dilation")
        
        
    
if __name__ == "__main__":
    main()