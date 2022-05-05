# CMSC630_Image_Analysis
Project for CMSC630 Image Analysis for low-, middle-, and high- level image analysis with filtering, classification and segmentation

## Install
clone the repository
install conda virtual environment
cd into root directory of git repository
`conda env create --file env.yaml`
`conda activate image`

## Run Part 1 : Filtering
`python main.py`

## Run Part 2 : Segmentation
`python part2.py`

## Run Part 3 : Classification
`python classification.py`

### If GPU available and not macos/arm64 architecture (not complete)
`python cuda.py`