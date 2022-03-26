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
    # OSError: cannot write mode F as JPEG
    # https://stackoverflow.com/questions/21669657/getting-cannot-write-mode-p-as-jpeg-while-operating-on-jpg-image
    # should convert be 'RGB' or 'L'
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

# edge detection algos
## sobel
# https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e
def sobel_edge_detector(img):
    M, N = img.shape
    
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], np.float32)
    
    copy_img = np.zeros((M, N))
    
    for i in range(1, M - 1):
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
    
 ## prewitt
def prewitt_edge_detector(img):
    M, N = img.shape
     
    Kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], np.float32)
    Ky = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], np.float32)
    
    copy_img = np.zeros((M, N))
    
    for i in range(1, M - 1):
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
    
### canny edge
# erosion
# dilation
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
        
    color_image = read_image(files[0])
    img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
    
    sobel = sobel_edge_detector(img)
    save_image(sobel, filenames[0], "_sobel")
    
    prewitt = prewitt_edge_detector(img)
    save_image(prewitt, filenames[0], "_prewitt")
    # canny = canny_edge_detector(img)
    
        
    
if __name__ == "__main__":
    main()