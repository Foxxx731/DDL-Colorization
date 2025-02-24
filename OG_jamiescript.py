import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import the regression version of the ECCV16 model from your colorization codebase
from colorizers.reg_model import eccv16  # Import the regression model

# -----------------------------
# Step 1: Define the Custom Dataset
# -----------------------------

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image file paths
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # Apply transformations (e.g., resizing)
        if self.transform:
            img = self.transform(img)

        # Convert to NumPy array
        img_np = np.array(img)

        # Convert from RGB to Lab color space
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize L channel to [-1, 1]
        img_lab[:, :, 0] = img_lab[:, :, 0] / 50.0 - 1.0

        # Normalize ab channels to [-2, 2] to increase saturation flexibility
        img_lab[:, :, 1:] = img_lab[:, :, 1:] / 55.0

        # Convert to tensor and rearrange dimensions to [C, H, W]
        img_lab = torch.from_numpy(img_lab.transpose((2, 0, 1)))

        # Split into L and ab channels
        L = img_lab[0:1, :, :]   # Shape: [1, H, W]
        ab = img_lab[1:3, :, :]  # Shape: [2, H, W]

        return L, ab

# -----------------------------
# Step 2: Define Data Transformations
# -----------------------------

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(20),  # Increased random rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Increased color jitter for more variation
])

# -----------------------------
# Step 3: Create Dataset Instances and DataLoaders
# -----------------------------

# Replace with your actual dataset paths
train_root = '/global/homes/l/liamfox/imagenet-mini-split/train'
val_root = '/global/homes/l/liamfox/imagenet-mini-split/val'

train_dataset = ColorizationDataset(root_dir=train_root, transform=data_transform)
val_dataset = ColorizationDataset(root_dir=val_root, transform=data_transform)

batch_size = 32  # Adjust based on your GPU memory

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -----------------------------
# Step 4: Load the Pre-trained Model and Set Up for Fine-Tuning
# -----------------------------

# Load the pre-trained ECCV16 colorization model
model = eccv16(pretrained=True)

# Move model to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable Data Parallelism
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for data parallelism.")
    model = nn.DataParallel(model)

model = model.to(device)

# Set model to training mode
model.train()

# -----------------------------
# Step 5: Define the Loss Function and Optimizer
# -----------------------------

criterion = nn.MSELoss()

# Use Adam optimizer with a lower learning rate for fine-tuning
learning_rate = 1e-6  # Lowered the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Step 6: Implement the Training and Validation Loops
# -----------------------------

def temperature_scaling(logits, temperature):
    return logits / temperature

num_epochs = 75  # Increased the number of epochs
temperature = 0.38  # Temperature parameter as per the paper

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    # Training Phase
    model.train()
    running_loss = 0.0

    for batch_idx, (L, ab) in enumerate(train_loader):
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()

        # Forward pass
        output_ab = model(L)

        # Apply temperature scaling to the output
        output_ab = temperature_scaling(output_ab, temperature)

        # Compute loss
        loss = criterion(output_ab, ab)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # Validation Phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for L_val, ab_val in val_loader:
            L_val, ab_val = L_val.to(device), ab_val.to(device)

            # Forward pass
            output_ab_val = model(L_val)

            # Apply temperature scaling to the output
            output_ab_val = temperature_scaling(output_ab_val, temperature)

            # Compute loss
            loss = criterion(output_ab_val, ab_val)
            val_loss += loss.item()

    val_epoch_loss = val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}')

    # Optionally, save the model checkpoint after each epoch
    #torch.save(model.state_dict(), f'eccv16_finetuned_epoch{epoch+1}.pth')

# -----------------------------
# Step 7: Save the Final Model
# -----------------------------

torch.save(model.state_dict(), 'eccv16_finetuned_final.pth')

# -----------------------------
# Step 8: Testing the Fine-Tuned Model
# -----------------------------

# Set the model to evaluation mode
model.eval()

# Path to your test image
test_image_path = '/pscratch/sd/l/liamfox/colorization/Groupdog_grayscaled.jpg'

# Function to test the model
def test_model(model, img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_gray = img.convert('L')  # Convert to grayscale
    img_gray = img_gray.resize((256, 256))
    img_np = np.array(img_gray)

    # Normalize L channel
    img_L = np.array(img_gray).astype(np.float32)
    img_L = img_L / 50.0 - 1.0
    L = torch.from_numpy(img_L).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]

    # Forward pass through the model
    with torch.no_grad():
        output_ab = model(L)

    # Apply temperature scaling to the output
    output_ab = temperature_scaling(output_ab, temperature)

    # Denormalize and concatenate with L channel
    output_ab = output_ab.cpu().numpy()[0]  # Shape: [2, H, W]
    output_ab = output_ab * 55.0  # Denormalize ab channels to [-110, 110] for more saturation
    L = L.cpu().numpy()[0][0]
    L = (L + 1.0) * 50.0  # Denormalize L channel

    # Merge channels and convert back to RGB
    img_lab_out = np.zeros((256, 256, 3))
    img_lab_out[:, :, 0] = L
    img_lab_out[:, :, 1:] = output_ab.transpose((1, 2, 0))
    img_rgb_out = cv2.cvtColor(img_lab_out.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Display or save the original and colorized images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title('Grayscale Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb_out)
    plt.title('Colorized Output')
    plt.axis('off')

    # Save the output image to a file
    plt.savefig('output_exp2.png')  # <-- This line will save the output to 'colorized_output.png'

# Test the model with your image
test_model(model, test_image_path)

