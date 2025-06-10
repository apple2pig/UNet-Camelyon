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
Execute the patch creation pipeline using:[utils/Gen_SegData.ipynb](https://github.com/apple2pig/UNet-Camelyon/blob/main/utils/Gen_SegData.ipynb)

# Data Post-Processing
1.py and 2.py are functions for:
Patch Prediction Processing (1.py): Generates heatmap from pathology patches
WSI Prediction Processing (2.py): Creates binary mask images from whole-slide images
