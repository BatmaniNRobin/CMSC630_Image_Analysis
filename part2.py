import os
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from main import (
    calc_histogram,
    read_yaml,
    read_image,
    convert_image_to_single_channel
    )

def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

# edge detection algos

## sobel and prewitt

# https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e

# following this: http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
# https://github.com/adeveloperdiary/blog/tree/master/Computer_Vision/Sobel_Edge_Detection
# results in a worse/darker version of prewitt/sobel using my convolve and is slower but less code

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
    
## canny
    # noise reduction
    # gradient calc
    # non-max suppression
    # double threshold
    # hysteresis - edge tracking

### creates 5x5 gaussian blur for canny - smoothens img to reduce noise
def gaussian_kernel(size):
    sigma = 1
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gauss_img = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    
    return gauss_img


# https://betterdatascience.com/implement-convolutions-from-scratch-in-python/

### applies filter to img
def convolve(img, kernel):
    M, N = img.shape
    kernel_row, kernel_col = kernel.shape
    
    copy_img = np.zeros((M - kernel_row + 1, N - kernel_col + 1))
    
    for row in tqdm(range(M - kernel_row + 1)):
        for col in range(N - kernel_col + 1):
            for i in range(kernel_row):
                for j in range(kernel_col):
                    value = img[row + i, col + j]
                    weight = kernel[i, j]
                    copy_img[row, col] += value * weight
                    
    return copy_img


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
def canny_edge_detector(img):
    gauss_img = gaussian_kernel(5) # 5x5 window
    
    blurred_img = convolve(img, gauss_img)
    
    sobel, theta = sobel_filters(blurred_img)
    
    suppression = non_max_suppression(sobel, theta)
    
    threshold_img, weak, strong = threshold(suppression)
    
    canny_img = hysteresis(threshold_img, weak, strong)
    
    return canny_img

## binary image
def binarize(img, threshold_value=127):
    color_1 = 255
    color_2 = 0
    initial_conv = np.where((img <= threshold_value), img, color_1)
    final_conv = np.where((initial_conv > threshold_value), initial_conv, color_2)
    
    return final_conv


# https://medium.com/@ami25480/morphological-image-processing-operations-dilation-erosion-opening-and-closing-with-and-without-c95475468fca

# dilation
def dilation(binary_img):
    M,N = binary_img.shape
    
    # structuring element ie. filter
    SE = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]])
    constant = 1
    
    copy_img = np.zeros((M,N), dtype=np.uint8)
    
    for i in range(1, M - 1):
      for j in range(1, N - 1):
            temp = binary_img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            value = temp * SE
            copy_img[i,j] = np.max(value)
    
    return copy_img


# https://python.plainenglish.io/image-dilation-explained-easily-e085c47fbac2
# NOT USED https://python.plainenglish.io/image-erosion-explained-in-depth-using-numpy-320c01b674a8

# erosion
def erosion(binary_img):
    M,N = binary_img.shape
    
    k = 3 # window size
    SE = np.ones((k,k), dtype=np.uint8) # structuring element
    constant = (k-1) // 2
    
    copy_img = np.zeros((M,N), dtype=np.uint8)
    
    for i in range(constant, M - constant):
      for j in range(constant,N - constant):
            tempVal = binary_img[i - constant:i + constant + 1, j - constant:j + constant + 1]
            value = tempVal * SE
            copy_img[i,j] = np.min(value)
    
    return copy_img


# 2 segmentation techniques

## hist thresholding
    # remove noise
    # calc hist
    # find hist split value/threshold
    # categorize into each side
    # return foreground

# https://theailearner.com/2019/07/19/balanced-histogram-thresholding/

def balance_hist(hist):
    # Starting point of histogram
    start = np.min(np.where(hist > 0))
    
    # End point of histogram
    end = np.max(np.where(hist > 0))
    
    # Center of histogram
    middle = (start + end) // 2
    
    # Left side weight
    weight_left = np.sum(hist[0:middle + 1])
    
    # Right side weight
    weight_right = np.sum(hist[middle + 1:end + 1])
    
    # Until starting point not equal to endpoint
    while (start != end):
        # If right side is heavier
        if (weight_right > weight_left):
            # Remove the end weight
            weight_right -= hist[end]
            end -= 1
            
            # Adjust the center position and recompute the weights
            if ((start + end) // 2) < middle:
                weight_left -= hist[middle]
                weight_right += hist[middle]
                middle -= 1
        else:
            # If left side is heavier, remove the starting weight
            weight_left -= hist[start]
            start += 1
            
            # Adjust the center position and recompute the weights
            if ((start + end) // 2) >= middle:
                weight_left += hist[middle + 1]
                weight_right -= hist[middle + 1]
                middle += 1
    return middle


def hist_threshold(img, hist):
    # calculates the threshold value
    middle = balance_hist(hist)
    
    copy_img = np.copy(img)
    
    # divide into foreground and background
    copy_img[copy_img >= middle] = 255
    copy_img[copy_img < middle] = 0
    
    copy_img = copy_img.astype(np.uint8).reshape(img.shape)
    
    return copy_img

## k means clustering

# calculate pairwise distance 
def getMin(pixel, centroids):
    minDist = 9999
    minIndex = 0
    
    for i in range(len(centroids)): # 2
        dist = np.sqrt(np.sum(np.square(pixel - centroids[i])))
        if(dist < minDist):
            minDist = dist
            minIndex = i
    
    return minIndex

# check if new centroids are converging towards clusters or not, output true or false
def converged(centroids, old_centroids):
    
    if(len(old_centroids) == 0):
        return False
    
    if(len(centroids) <= 5):
        a = 1
    # elif(len(centroids) <= 10):
    #     a = 2
    # else:
    #     a = 4
    
    for i in range(len(centroids)):
        cent = centroids[i]
        old_cent = old_centroids[i]
        
        if ((int(old_cent) - a) <= cent <= (int(old_cent) + a)):
            continue
        else:
            return False
    
    return True

# calclulate k means and return 2 centroids
def k_means(hist, k):
    centroids = []
    clusters = {}
    
    
    # set k centroids randomly
    for _ in range(k):
        cent = np.random.randint(0, len(hist))
        centroids.append(cent) 
        
    centroids = np.asarray(centroids)
    old_centroids = np.zeros(2)
    i = 1
    
    # do 5 iterations, guide uses 20 but for 2 clusters 5 is enough probably
    while ( i<= 5 and not converged(centroids, old_centroids)):
        i += 1
        old_centroids = np.copy(centroids)
        
        # calculate pairwise distance
        for x in range(len(hist)):
            p = hist[x]
            minIndex = getMin(hist[x], centroids)
            try:
                clusters[minIndex].append(p)
            except KeyError:
                clusters[minIndex] = [p]
        
        # adjust centroids
        new_centroids = np.array(centroids)
        keys = sorted(clusters.keys())
        
        for key in keys:
            n = np.mean(clusters[key])
            new_centroids = new_centroids + int(n)

        centroids = new_centroids
        
    return centroids
        

## clustering - k-means (bonus points for other ones)
def kMeans_clustering(img, hist, k):
    kmeans = k_means(hist, k)
    
    copy_img = np.copy(img)
    
    difference = abs(kmeans[1] - kmeans[0])
    
    # divide into foreground and background
    copy_img[copy_img >= difference] = 255
    copy_img[copy_img < difference] = 0
    
    copy_img = copy_img.astype(np.uint8).reshape(img.shape)
    
    return copy_img



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
        
        # format cli output
        print(filenames[i])
        
        color_image = read_image(files[i])
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        
        sobel = sobel_or_prewitt(img, "sobel")
        save_image(sobel, filenames[i], "_sobel")
        
        prewitt = sobel_or_prewitt(img, "prewitt")
        save_image(prewitt, filenames[i], "_prewitt")

        canny = canny_edge_detector(img)
        save_image(canny, filenames[i], "_canny")
        
        binary_img = binarize(img)
        save_image(binary_img, filenames[i], "_binary")
        
        # erosion and dilation on binary images
        eroded = dilation(binary_img)
        save_image(eroded, filenames[i], "_erosion")
        
        dilated = erosion(binary_img)
        save_image(dilated, filenames[i], "_dilation")
        
        # erosion and dilation on edge map
        edge_dilated = dilation(canny)
        edge_dilated_twice = dilation(edge_dilated)
        save_image(edge_dilated, filenames[i], "_edge_dilation")
        
        ### because canny is 1 pixel wide this just erases everything
        # edge_eroded = erosion(canny)
        # save_image(edge_eroded, filenames[i], "_edge_erosion_canny")
        
        edge_eroded_dilation = erosion(edge_dilated)
        save_image(edge_eroded_dilation, filenames[i], "_edge_erosion_after_dilation")
        
        edge_eroded_dilation_twice = erosion(edge_dilated_twice)
        save_image(edge_eroded_dilation_twice, filenames[i], "edge_eroded_after_twice_dilated")
        
        img_hist = calc_histogram(img)
        
        histogram_threshold = hist_threshold(img, img_hist)
        save_image(histogram_threshold, filenames[i], "_hist_threshold")
        
        clustering = kMeans_clustering(img, img_hist, safe_conf["K_VALUE"])
        save_image(clustering, filenames[i], "_kMeans_clustering")
        
    
if __name__ == "__main__":
    main()