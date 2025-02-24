import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from base_color import *  # Make sure base_color.py is available
import torch.utils.model_zoo as model_zoo


# ECCV16 Generator class
class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), norm_layer(64)]
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), norm_layer(128)]
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU(True), norm_layer(256)]
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(512)]
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), norm_layer(512)]
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), nn.ReLU(True), norm_layer(512)]
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), norm_layer(512)]
        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU(True), nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))


def eccv16(pretrained=True):
    model = ECCVGenerator()
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth', map_location='cpu', check_hash=True))
    return model


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

batch_size = 16  # Adjust based on your GPU memory

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# -----------------------------
# Step 4: Load the Pre-trained Model and Set Up for Fine-Tuning
# -----------------------------

# Split model components across GPUs
device0 = torch.device('cuda:0')  # First GPU
device1 = torch.device('cuda:1')  # Second GPU

# Assume eccv16 has two submodules: encoder and decoder
class ModelParallelColorization(nn.Module):
    def __init__(self, base_model):
        super(ModelParallelColorization, self).__init__()
        self.encoder = base_model.encoder.to(device0)  # Place encoder on device0
        self.decoder = base_model.decoder.to(device1)  # Place decoder on device1

    def forward(self, x):
        x = x.to(device0)
        x = self.encoder(x)  # Forward pass on encoder
        x = x.to(device1)    # Move intermediate result to device1
        x = self.decoder(x)  # Forward pass on decoder
        return x

# Load the pre-trained ECCV16 colorization model
base_model = eccv16(pretrained=True)
model = ModelParallelColorization(base_model)

# Ensure model is in training mode
model.train()

# -----------------------------
# Step 5: Define the Loss Function and Optimizer
# -----------------------------

criterion = nn.MSELoss()

# Use Adam optimizer with a lower learning rate for fine-tuning
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Step 6: Implement the Training and Validation Loops
# -----------------------------

for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    # Training Phase
    model.train()
    running_loss = 0.0

    for batch_idx, (L, ab) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        L = L.to(device0)  # Move L channel to the first GPU
        ab = ab.to(device1)  # Move ab channels to the second GPU

        output_ab = model(L)

        # Compute loss
        loss = criterion(output_ab, ab)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # Validation Phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for L_val, ab_val in val_loader:
            L_val = L_val.to(device0)
            ab_val = ab_val.to(device1)

            # Forward pass
            output_ab_val = model(L_val)

            # Compute loss
            loss = criterion(output_ab_val, ab_val)
            val_loss += loss.item()

    val_epoch_loss = val_loss / len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), f'eccv16_finetuned_epoch{epoch+1}.pth')

# -----------------------------
# Step 7: Save the Final Model
# -----------------------------

torch.save(model.state_dict(), 'eccv16_finetuned_final.pth')

# -----------------------------
# Step 8: Testing the Fine-Tuned Model
# -----------------------------

def test_model(model, img_path):
    model.eval()
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_gray = img.convert('L')
    img_np = np.array(img_gray)

    # Normalize and prepare L channel
    img_L = (np.array(img_gray) / 50.0 - 1.0).astype(np.float32)
    L = torch.from_numpy(img_L).unsqueeze(0).unsqueeze(0).to(device0)

    with torch.no_grad():
        output_ab = model(L)

    # Denormalize and process
    output_ab = output_ab.cpu().numpy()[0] * 55.0
    L = (L.cpu().numpy()[0][0] + 1.0) * 50.0

    img_lab = np.zeros((256, 256, 3))
    img_lab[:, :, 0] = L
    img_lab[:, :, 1:] = output_ab.transpose((1, 2, 0))
    img_rgb = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Grayscale")

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.title("Colorized")
    plt.show()

# Path to your test image
test_image_path = '/path/to/your/test_image.jpg'

# Call the testing function
test_model(model, test_image_path)