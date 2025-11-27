import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from main import SegNet  # Import the class from your main script

# 1. CONFIGURATION
# Path to your trained model
MODEL_PATH = 'segnet_final.pth'
# Path to an image you want to test (pick one from your val or test folder)
TEST_IMAGE_PATH = './CamVid/test/0001TP_008550.png'
# CamVid Class Colors (Same as in training)
class_colors = [
    (128, 128, 128),  # 0: Sky
    (128, 0, 0),      # 1: Building
    (192, 192, 128),  # 2: Pole
    (128, 64, 128),   # 3: Road
    (0, 0, 192),      # 4: Pavement
    (128, 128, 0),    # 5: Tree
    (192, 128, 128),  # 6: SignSymbol
    (64, 64, 128),    # 7: Fence
    (64, 0, 128),     # 8: Car
    (64, 64, 0),      # 9: Pedestrian
    (0, 128, 192),    # 10: Bicyclist
]

def decode_segmap(image, nc=11):
    """
    Converts a tensor of class IDs (0-10) back into an RGB image for visualization.
    """
    # Create empty RGB array
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    # Assign colors
    for l in range(0, nc):
        idx = image == l
        r[idx] = class_colors[l][0]
        g[idx] = class_colors[l][1]
        b[idx] = class_colors[l][2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Model Architecture
    print("Loading model...")
    model = SegNet(input_channels=3, output_classes=11).to(device)
    
    # 2. Load the Trained Weights
    # map_location ensures it works even if you trained on GPU but test on CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Important: Sets BatchNorm and Dropout to evaluation mode
    
    # 3. Prepare the Image
    input_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    original_size = input_image.size
    
    # Resize to training resolution (360x480)
    # SegNet requires specific dimensions to handle the pooling/unpooling indices correctly
    resize_transform = transforms.Resize((360, 480))
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = resize_transform(input_image)
    input_tensor = tensor_transform(image).unsqueeze(0).to(device) # Add batch dim [1, 3, 360, 480]
    
    # 4. Run Prediction
    print("Running prediction...")
    with torch.no_grad():
        output = model(input_tensor)
        
    # Output shape is [1, 11, 360, 480]. 
    # We take the max across dim 1 to get the class ID for each pixel.
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # 5. Decode and Visualize
    rgb_mask = decode_segmap(pred_mask)
    
    # Optional: Resize back to original image size if you want
    # rgb_mask = np.array(Image.fromarray(rgb_mask).resize(original_size, Image.NEAREST))

    # Display side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    
    ax1.imshow(input_image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    ax2.imshow(rgb_mask)
    ax2.set_title('Predicted Segmentation')
    ax2.axis('off')
    
    plt.show()
    
    # Optional: Save result
    # Image.fromarray(rgb_mask).save('result.png')
    print("Done! Check the plot.")

if __name__ == "__main__":
    run_inference()