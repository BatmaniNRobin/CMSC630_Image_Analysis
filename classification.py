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

from part2 import (
    hist_threshold,
    balance_hist,
    erosion,
    dilation
)

def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")

def shrinking_and_growing(hist_threshold):
    
    eroded = erosion(hist_threshold)
    dilated = dilation(hist_threshold)
    
    sng = dilation(eroded)
    
    return sng, eroded, dilated


# https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated
def entropy(img, hist):
    
    # marg calculates marginal dist, then filter out probs == 0
    marg = hist / img.size
    marg = np.array(list(filter(lambda p: p > 0, marg)))
    
    # then sum whats left
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))

    return entropy

def area(sng):
    unique, counts = np.unique(sng, return_counts=True)
    counter = dict(zip(unique, counts))

    black_pixel_count = counter[0]

    return black_pixel_count


def extract_features(img, hist, sng):
    
    # calculate the randomness of the pixel intensities
    x1 = entropy(img, hist)
    
    # calculate mean of img histogram
    x2 = np.mean(hist)
    
    # calculate area
    x3 = area(sng)
    
    return {
        "features": [x1, x2, x3]
    }







def main():
    
    global safe_conf
    config_file = "config.yaml"
    safe_conf = read_yaml(config_file)
    
    data_loc = Path(safe_conf['DATA_DIR'])
    Path(safe_conf["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    
    files = list(data_loc.iterdir())
    filenames = [i for i in range(len(files))]
    
    features = []
    
    for i in range(len(files)):
        filenames[i] = os.path.basename(files[i])
        
        if (".BMP" in filenames[i]):
            filenames[i] = os.path.splitext(filenames[i])[0]
        
        # format cli output
        print(filenames[i])
        
        
        color_image = read_image(files[i])
        img = convert_image_to_single_channel(color_image, safe_conf['SELECTED_COLOR_CHANNEL'])
        
        img_hist = calc_histogram(img)
        
        histogram_threshold = hist_threshold(img, img_hist)
        save_image(histogram_threshold, filenames[i], "_hist_threshold")
        
        sng, eroded, dilated = shrinking_and_growing(histogram_threshold)
        ## TODO dont need erosion and dilation just testing them
        save_image(eroded, filenames[i], "_erosion")
        save_image(dilated, filenames[i], "_dilation")
        save_image(sng, filenames[i], "_sng")
        
        img_features = extract_features(img, img_hist, sng)
        
        # TODO append features and create a dataset
        
        
        
        
    
if __name__ == "__main__":
    main()