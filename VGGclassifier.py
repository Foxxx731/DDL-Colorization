from torchvision import models, transforms
from PIL import Image
import torch

# Load pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()  # Set the model to evaluation mode

# Define image transformations: resize, crop, normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the colorized image
input_image = Image.open('Colorized_Dog_eccv16.png')
input_image = input_image.convert("RGB")  # Convert to RGB if necessary
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Run the image through the classifier
with torch.no_grad():  # Disable gradient calculation for inference
    output = vgg16(input_batch)

# Get the top-1 predicted class
_, predicted_idx = torch.max(output, 1)
print(f"Predicted class index: {predicted_idx.item()}")

# Optional: If you want the class labels, load them
from torchvision.models import vgg
labels = vgg.VGG16_Weights.DEFAULT.meta["categories"]
print(f"Predicted class label: {labels[predicted_idx.item()]}")