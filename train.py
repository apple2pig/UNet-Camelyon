# Import required libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
from tqdm import tqdm  # Progress bar visualization
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific warnings

# Define training function
def train(epochs):
    """
    Train the model for specified number of epochs
    Args:
        epochs (int): Number of training iterations
    Returns:
        model: Trained model
    """
    
    # Initialize best metric trackers
    best_acc = float('-inf')   # Best accuracy
    best_dice = float('-inf')  # Best Dice coefficient
    best_iou = float('-inf')   # Best Intersection-over-Union
    best_auc = float('-inf')   # Best Area Under Curve

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        dt_size = len(train_dataloaders)  # Number of batches
        epoch_loss = 0  # Accumulated loss per epoch
        step = 0        # Batch counter
        
        # Initialize progress bar
        with tqdm(total=dt_size, desc=f'Train E{epoch+1}', colour='blue') as pbar:
            # Iterate through training batches
            for x, y in train_dataloaders:
                step += 1
                # Move data to GPU if available
                inputs = x.to(device)
                labels = y.to(device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                # Optional gradient clipping (commented out)
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=0.7)
                optimizer.step()
                
                # Update loss tracking
                epoch_loss += loss.item()
                
                # Update progress bar with current loss
                pbar.set_postfix(**{"Loss": f"{epoch_loss/step:.4f}"})
                pbar.update(1)  # Update progress
            
            pbar.close()  # Close progress bar after epoch

            # Calculate validation metrics
            Acc, Dice, Iou, auc = calculate_acc(model, val_dataloaders, device)

            # Update best metrics
            best_acc = max(Acc, best_acc)
            best_dice = max(Dice, best_dice)
            best_auc = max(auc, best_auc)

            # Save model if Iou improves
            if Iou > best_iou:
                best_iou = Iou
                torch.save(model.state_dict(), weights_path)
                print('Model saved!')
                
            # Optional periodic saving (commented out)
            # if epoch%5==0:
            #     torch.save(model.state_dict(), f'{weights_path}/{epoch}.pth')
    
    return model  # Return trained model


# Main execution block
if __name__ == '__main__':
    # Import custom modules
    from utils.read_data import LiverDataset      # Dataset loader
    from utils.evaluate import calculate_acc      # Evaluation metrics
    from UNet import Unet                         # Model architecture

    # Set device configuration (use GPU if available)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Initialize U-Net model (3 input channels, 3 output classes)
    model = Unet(3, 3).to(device)
    
    # Training hyperparameters
    Batch_Size = 6    # Samples per batch
    Epochs = 200      # Total training epochs
    
    # Import segmentation models library for loss function
    import segmentation_models_pytorch as smp
    # Define composite loss (Binary Cross-Entropy + Dice Loss)
    criterion = smp.utils.losses.BCELoss() + smp.utils.losses.DiceLoss()
    
    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Model checkpoint path
    weights_path = 'UNet_17.pth'
    
    # Load pre-trained weights if available
    if os.path.isfile(weights_path):
        model.load_state_dict(torch.load(weights_path))

    # Create dataset objects
    train_dataset = LiverDataset(mode='train', dn='Camelyon16')  # Training dataset
    val_dataset = LiverDataset(mode='val', dn='Camelyon16')      # Validation dataset
    
    # Create data loaders
    train_dataloaders = DataLoader(
        train_dataset,
        batch_size=Batch_Size,
        shuffle=True,      # Shuffle training data
        num_workers=4      # Parallel data loading
    )
    val_dataloaders = DataLoader(
        val_dataset,
        batch_size=Batch_Size,
        shuffle=True,      # Shuffle validation data
        num_workers=4
    )

    # Start training process
    train(Epochs)