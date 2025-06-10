# Import required libraries
import openslide  # For reading whole-slide images (WSI)
import torch
from PIL import Image  # Image processing
from tqdm import tqdm  # Progress bar visualization
import numpy as np
from torchvision import transforms  # Image transformations
import cv2  # Computer vision operations

def create_heatmap(pre, ima):
    """
    Create and save a heatmap visualization overlay
    Args:
        pre: Prediction array (2D numpy array)
        ima: Original PIL Image for overlay
    """
    # Normalize prediction values to 0-255 range
    normalized_pre = cv2.normalize(pre, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Apply color mapping (jet colormap) to predictions
    heatmap = cv2.applyColorMap(np.uint8(normalized_pre), cv2.COLORMAP_JET)
    
    # Convert from BGR to RGB color space
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create smoothing kernel and apply to heatmap
    kernel = np.array(np.ones((5, 5)), dtype=np.uint8)
    heatmap = cv2.filter2D(heatmap, -1, kernel)
    
    # Convert original image to numpy array
    ima = np.asarray(ima)
    
    # Blend heatmap with original image (60% heatmap + 40% image)
    heatmap_and_image = cv2.addWeighted(heatmap, 0.6, ima, 0.4, 0)
    
    # Convert back to PIL Image format
    heatmap = Image.fromarray(heatmap_and_image)
    
    # Resize to 10% of original dimensions for manageable output
    width, height = int(heatmap.size[0] * 0.1), int(heatmap.size[1] * 0.1)
    heatmap = heatmap.resize((width, height))

    # Save final heatmap visualization
    heatmap.save('UNet_pre_test40.png')

def val_wsi(image, hw, batch_size):
    """
    Process Whole Slide Image (WSI) using sliding window approach
    Args:
        image: OpenSlide image object
        hw: Patch size (height/width)
        batch_size: Processing batch size (unused in current implementation)
    """
    # Calculate processing dimensions (round down to nearest multiple of patch size)
    size = [image.dimensions[0] // hw * hw, image.dimensions[1] // hw * hw]
    
    # Initialize output array for predictions
    output = np.zeros([size[1], size[0]], dtype=np.uint8)
    
    # Calculate total patches for progress bar (only processes 30% of vertical dimension)
    total_patches = size[0] * (0.3 * size[1]) / (hw * hw)
    
    # Initialize progress bar
    with tqdm(total=total_patches, colour="black", desc="Processing") as pbar:
        # Slide window horizontally
        for a in range(0, size[0], hw):
            # Slide window vertically (only from 55,400 to 121,448 pixels)
            for b in range(0, size[1], hw):
                # Extract image patch at current position
                patch = image.read_region(location=(a, b), size=(hw, hw), level=0).convert('RGB')
                
                # Apply transformations and prepare for model input
                x = x_transforms(patch).unsqueeze(0).to(device)
                
                # Model prediction (no gradient calculation)
                with torch.no_grad():
                    pred = model(x)[0]  # Take first output (assumes batch size 1)
                
                # Get maximum prediction across channels
                pred = torch.max(pred, dim=0).values
                pred = pred.cpu().numpy()
                
                # Convert predictions to 0-255 range
                tem_output = (pred * 255).astype(np.uint8)
                
                # Insert prediction into output array
                output[b:b + hw, a:a + hw] = tem_output
                
                # Update progress bar
                pbar.update(1)
        pbar.close()

    # Create full-scale image region for overlay
    full_region = image.read_region(location=(0, 0), size=size, level=0).convert('RGB')
    
    # Generate and save heatmap visualization
    create_heatmap(output, full_region)

# Main execution block
if __name__ == '__main__':
    # Remove PIL image size limitation
    Image.MAX_IMAGE_PIXELS = None
    
    # Set device configuration (use GPU if available)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Define image preprocessing pipeline
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL to tensor [0,1]
        transforms.Normalize(  # Normalize to [-1,1] range
            [0.5, 0.5, 0.5], 
            [0.5, 0.5, 0.5]
        )
    ])

    # Import model architecture
    from UNet import Unet
    
    # Initialize U-Net model (3 input channels, 3 output classes)
    model = Unet(3, 3).to(device)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load('UNet.pth'))
    
    # Set model to evaluation mode
    model.eval()

    # Load whole-slide image (WSI)
    image = openslide.OpenSlide('/Camelyon16/test_040.tif')
    
    # Process WSI with 512x512 patches
    val_wsi(image, 512, batch_size=12)