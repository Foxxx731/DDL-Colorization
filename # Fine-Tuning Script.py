# Fine-Tuning Script
import torch
import torch.optim as optim
import torch.nn as nn
from eccv16 import eccv16
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np

def extract_ab_channels(inputs):
    # Convert the grayscale tensor to a format suitable for OpenCV
    inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (batch, height, width, channels)
    inputs_lab = []
    for img in inputs_np:
        # Expand single-channel grayscale to 3-channel BGR
        pseudo_bgr = cv2.merge([img, img, img])
        # Convert to LAB color space
        lab = cv2.cvtColor((pseudo_bgr * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        inputs_lab.append(lab[:, :, 1:])  # Extract AB channels
    inputs_ab = np.array(inputs_lab)  # Convert list to numpy array
    inputs_ab = torch.from_numpy(inputs_ab).permute(0, 3, 1, 2).to(inputs.device).float() / 255.0
    return inputs_ab


# Load the model
model = eccv16(pretrained=True).train()  # Start with pretrained model for fine-tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Small learning rate for fine-tuning
criterion = nn.MSELoss()

# Load the dataset
train_dataset = datasets.ImageFolder(
    '/global/homes/l/liamfox/imagenet-mini-split/train',  # Adjusted with your dataset path
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),  # To match model input of L channel
        transforms.ToTensor()
    ])
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 20  # Increase number of epochs for a better training session
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs_ab = model(inputs)

        # Convert inputs to AB channels
        inputs_ab = extract_ab_channels(inputs)

        # Calculate loss and backward pass
        loss = criterion(outputs_ab, inputs_ab)  # Compare outputs with AB channels
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item()}")

    print(f"Finished Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_models/eccv16_finetuned.pth')
