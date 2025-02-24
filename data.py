import os
import shutil

# Define the paths
dataset_root = '/global/homes/l/liamfox/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini'
output_dir = '/global/homes/l/liamfox/imagenet-mini-split'

# Create separate train and validation directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to copy images into the train or validation directories
def copy_images(source_dir, target_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Ensure the file is an image
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                # Maintain subdirectory structure
                rel_dir = os.path.relpath(root, source_dir)
                target_sub_dir = os.path.join(target_dir, rel_dir)
                os.makedirs(target_sub_dir, exist_ok=True)

                # Copy the file to the target directory
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_sub_dir, file)
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"Copied: {source_file} to {target_file}")
                except Exception as e:
                    print(f"Failed to copy {source_file} to {target_file}: {e}")

# Assume validation data exists in the 'val' subdirectory of the dataset root
print("Copying validation data...")
copy_images(os.path.join(dataset_root, 'val'), val_dir)

# Optional: If the dataset has no train/val split, duplicate val data to train
print("Copying training data...")
copy_images(os.path.join(dataset_root, 'val'), train_dir)

print("Copying complete.")
