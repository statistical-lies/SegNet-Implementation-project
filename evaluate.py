import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from main import SegNet, CamVidDataset  # Import from your main script

# --- CONFIGURATION ---
MODEL_PATH = 'segnet_final.pth'
TXT_TEST = './CamVid/test.txt'  # Make sure you have this file
ROOT_DIR = '.'
BATCH_SIZE = 4
NUM_CLASSES = 11

def compute_iou(model, loader, device):
    model.eval()
    
    # Confusion matrix to store counts: Rows=GT, Cols=Pred
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    print("Starting evaluation on Test Set...")
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device) # Shape: [Batch, H, W]
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # Shape: [Batch, H, W]
            
            # Move to CPU for numpy calculation
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            
            # Flatten to 1D arrays
            preds_flat = preds.flatten()
            masks_flat = masks.flatten()
            
            # Filter out "Void" class (11 or -100)
            # We only care about pixels where the Ground Truth is a valid class (0-10)
            valid_indices = np.where((masks_flat >= 0) & (masks_flat < NUM_CLASSES))
            preds_flat = preds_flat[valid_indices]
            masks_flat = masks_flat[valid_indices]
            
            # Update Confusion Matrix
            # This is a fast trick to update the matrix
            # bincount calculates occurrences of each (label, pred) pair
            n = NUM_CLASSES
            indices = masks_flat * n + preds_flat
            count = np.bincount(indices, minlength=n**2)
            confusion_matrix += count.reshape(n, n)
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1} batches...")

    # --- CALCULATE METRICS ---
    
    # 1. Global Accuracy: Sum of Diagonal / Sum of Total
    # (Correct Pixels) / (Total Pixels)
    global_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    
    # 2. Class Accuracy: Diagonal / Sum of Rows
    # (Correct for Class X) / (Total True Pixels of Class X)
    # We use nan_to_num to handle classes that might not appear in the test set
    class_acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    mean_class_acc = np.nanmean(class_acc)
    
    # 3. Mean IoU: Diagonal / (Sum of Rows + Sum of Cols - Diagonal)
    # Intersection / Union
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou = intersection / union
    mean_iou = np.nanmean(iou)
    
    return global_acc, mean_class_acc, mean_iou, class_acc, iou

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Load Data
    test_dataset = CamVidDataset(TXT_TEST, ROOT_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load Model
    model = SegNet(input_channels=3, output_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # Run Evaluation
    g_acc, c_acc, mIoU, per_class_acc, per_class_iou = compute_iou(model, test_loader, device)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Global Accuracy (G):      {g_acc*100:.2f}%")
    print(f"Class Average Acc (C):    {c_acc*100:.2f}%")
    print(f"Mean IoU (mIoU):          {mIoU*100:.2f}%")
    print("-" * 30)
    print("PER CLASS IoU:")
    
    class_names = ['Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree', 
                   'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist']
                   
    for i, name in enumerate(class_names):
        print(f"{name:12s}: {per_class_iou[i]*100:.2f}%")