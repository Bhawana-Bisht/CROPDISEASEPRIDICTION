import os
import shutil
import random

random.seed(42)

# Source: Final dataset created earlier
source_dir = r"C:\Users\bhawa\OneDrive\Desktop\MINOR PROJECT\CropDiseasePrediction\data\rice leaf diseases dataset\FINAL_7000_DATASET"

# Output folders
base_output = r"C:\Users\bhawa\OneDrive\Desktop\MINOR PROJECT\CropDiseasePrediction\data"
train_dir = os.path.join(base_output, "train")
val_dir = os.path.join(base_output, "val")
test_dir = os.path.join(base_output, "test")

# Class Names
classes = ["Bacterialblight", "Brownspot", "Leafsmut", "HealthyLeaf"]

# Remove old splits if they exist
for folder in [train_dir, val_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Re-create folder structure
for folder in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# Function to copy files
def copy_files(file_list, dest_folder, class_name):
    for f in file_list:
        shutil.copy(f, os.path.join(dest_folder, class_name))

# Split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

print("\n📌 Starting Dataset Split...\n")

# Process each class
for cls in classes:
    class_path = os.path.join(source_dir, cls)

    # List all image paths
    images = [
        os.path.join(class_path, img)
        for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split into train/val/test
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Copy files
    copy_files(train_files, train_dir, cls)
    copy_files(val_files, val_dir, cls)
    copy_files(test_files, test_dir, cls)

    print(f"✔ {cls}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

print("\n🎉 Dataset Split Completed Successfully!\n")
print(f"📁 Train Folder: {train_dir}")
print(f"📁 Val Folder:   {val_dir}")
print(f"📁 Test Folder:  {test_dir}")
