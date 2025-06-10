This is a project that applies a semantic segmentation network to process Whole Slide Images (WSI). The example dataset used is the Camelyon17 dataset, which can be downloaded from: https://camelyon17.grand-challenge.org/Data/

# File Structure & Path Configuration
To run this project, please replace all code paths with your actual file paths. 
The expected directory structure is:  
├── Camelyon16/  
│   ├── train/  
│   │   ├── img/  
│   │   ├── mask/  
│   ├── val/  
│   │   ├── img/  
│   │   ├── mask/  



# Patch Generation & Training Preparation
Execute the patch creation pipeline using:[Gen_SegData.ipynb](https://github.com/apple2pig/UNet-Camelyon/blob/main/utils/Gen_SegData.ipynb)

# Data Post-Processing
[Pre_patches.py](https://github.com/apple2pig/UNet-Camelyon/blob/main/pre_patches.py) and [Pre_WSI.py](https://github.com/apple2pig/UNet-Camelyon/blob/main/pre_WSI.py) are functions for:
Patch Prediction Processing ([Pre_patches.py](https://github.com/apple2pig/UNet-Camelyon/blob/main/pre_patches.py)): Generates heatmap from pathology patches
WSI Prediction Processing ([Pre_WSI.py](https://github.com/apple2pig/UNet-Camelyon/blob/main/pre_WSI.py)): Creates binary mask images from whole-slide images
<img src="https://github.com/user-attachments/assets/90a7b38a-9ee3-4cd7-b493-200268dd7a1a" width="450" height="300">
<img src="https://github.com/user-attachments/assets/e7ef4292-6b87-463e-a958-a3a9b7bfa649" width="550" height="300">


