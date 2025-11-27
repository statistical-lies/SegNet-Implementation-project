import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from model import SegNet
from dataset import CamVidDataset
from utils import calculate_class_weights

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_segnet():
    # --- CONFIGURATION ---
    
    # 1. Paths to the text files (Inside the CamVid folder)
    TXT_TRAIN = './CamVid/train.txt'
    TXT_VAL = './CamVid/val.txt'
    
    # 2. The root directory for image loading
    # We use '.' because the text file paths (after cleaning) will start with "CamVid/"
    # So os.path.join('.', 'CamVid/train/...') is the correct path.
    ROOT_DIR = '.' 
    
    # 3. Hyperparameters
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    BATCH_SIZE = 4       # Set to 4 or 5 to match the paper's batch size
    NUM_EPOCHS = 100 
    NUM_CLASSES = 11
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- DATA ---
    print("Initializing Datasets from text files...")
    
    # Pass the text file path and the root directory
    train_dataset = CamVidDataset(TXT_TRAIN, ROOT_DIR)
    val_dataset = CamVidDataset(TXT_VAL, ROOT_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- MODEL ---
    model = SegNet(input_channels=3, output_classes=NUM_CLASSES).to(device)

    # --- LOSS & OPTIMIZER ---
    # CRITICAL: Use the hardcoded weights from the Caffe prototxt file you uploaded
    # These are the exact values used by the authors [cite: 48]
    weights_list = [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 
                    9.6446, 1.8418, 0.6823, 6.2478, 7.3614]
    
    class_weights = torch.FloatTensor(weights_list).to(device)
    print(f"Using manual paper weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

     # --- TRAINING LOOP ---
    starting_time = time.time()
    print(f'Starting training...at {starting_time}')
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calculate Loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        duration = time.time() - start_time
        
        print(f"End of Epoch {epoch+1} | Time: {duration:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint every 10 epochs or if it's the best model
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'segnet_epoch_{epoch+1}.pth')
            print(f"Model saved: segnet_epoch_{epoch+1}.pth")

    print(f"Training Complete at {time.time() - starting_time}")
    torch.save(model.state_dict(), 'segnet_final.pth')

if __name__ == "__main__":
    train_segnet()