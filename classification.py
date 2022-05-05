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
    balance_hist
)

def save_image(img, filename, applied_method):
    new_img = Image.fromarray(img).convert("L")
    new_img.save(safe_conf["OUTPUT_DIR"] + filename + applied_method + ".jpg", format="JPEG")


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
        
        img_hist = calc_histogram(img)
        
        histogram_threshold = hist_threshold(img, img_hist)
        save_image(histogram_threshold, filenames[i], "_hist_threshold")
        
        
        
        
    
if __name__ == "__main__":
    main()