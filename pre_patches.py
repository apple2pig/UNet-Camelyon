# Import required libraries
import torch
from PIL import Image  # Image processing
import matplotlib.pyplot as plt  # Plotting and visualization
import numpy as np
import matplotlib.colors as colors  # Custom color maps
from matplotlib import cm  # Colormap utilities

# Define image preprocessing pipeline
from torchvision import transforms
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL to tensor [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
])

def pre2heatmap(img_path, save_path):
    """
    Generate and save prediction heatmap overlays
    Args:
        img_path: Path to input image
        save_path: Directory to save results
    """
    with torch.no_grad():  # Disable gradient calculation
        # Load and process input image
        img = Image.open(img_path).convert('RGB')
        size = img.size[0]  # Assume square image
        
        # Prepare image for model input
        img2 = x_transforms(img).to(device)
        x = img2.unsqueeze(0)  # Add batch dimension
        
        # Model prediction
        y = model(x)
        predicted = y.cpu().numpy()[0]  # Convert to numpy array
        
        # Create 2D output by taking max across channels
        output = np.zeros((size, size))
        for p in range(size):
            for j in range(size):
                # Take maximum value across all channels
                max_value = np.maximum(predicted[0][p, j], 
                                      np.maximum(predicted[1][p, j], 
                                      predicted[2][p, j]))
                output[p, j] = max_value

        # Create rainbow colormap
        cmap = colors.ListedColormap(cm.rainbow(np.linspace(0, 1, 256)))
        
        # Generate and save heatmap visualization
        plt.imshow(output, cmap=cmap)
        plt.axis('off')  # Hide axes
        plt.savefig(f'{save_path}heatmap.png', bbox_inches='tight', pad_inches=0)
        
        # Create overlay with original image
        img = Image.open(img_path).convert('RGBA')
        heatmap = Image.open(f'{save_path}heatmap.png').convert('RGBA')
        heatmap = heatmap.resize(img.size)  # Match sizes
        superimposed_image = Image.blend(img, heatmap, 0.4)  # 40% overlay
        superimposed_image.save(save_path + img_path.split('/')[-1])  # Save result


# --- Model Initialization Section ---
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # Set device

# Load pre-trained U-Net model
from UNet import Unet
model = Unet(3, 3).to(device)  # 3-channel input, 3-class output
model.load_state_dict(torch.load('UNet.pth'))  # Load weights
model.eval()  # Set to evaluation mode


# --- Batch Processing Section ---
import glob
from tqdm import tqdm  # Progress bar
import os

# Configure input/output paths
patches = 'patch_path/'  # Directory with input images
num_patches = len(os.listdir(patches))  # Count files to process
save_to = 'save_path/'  # Output directory

# Process all images in directory
with tqdm(total=num_patches, desc='Processing', colour='blue') as pbar:
    for img in glob.glob(patches + '*.png'):  # Find all PNG files
        pre2heatmap(img, save_to)  # Generate heatmap
        pbar.update(1)  # Update progress
    pbar.close()