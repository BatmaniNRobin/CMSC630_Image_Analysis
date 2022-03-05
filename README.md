# CMSC630_Image_Analysis
Project for CMSC630 Image Analysis for low-, middle-, and high- level image analysis with filtering, classification and segmentation

## Install
clone the repository
install conda virtual environment
cd into root directory of git repository
`conda env create --file env.yaml`
`conda activate image`

## Run
`python main.py`

### If GPU available and not macos/arm64 architecture
`python cuda.py`