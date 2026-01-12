import os
from PIL import Image
from torchvision import transforms

# Folder paths
healthy_folder = r"data\rice_leaf_diseases\Healthy Rice Leaf"
augmented_folder = r"data\rice_leaf_diseases\Healthy_Rice_Augmented"

os.makedirs(augmented_folder, exist_ok=True)

# Define augmentation transforms
augment = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

# Generate augmented images
count = 0
target_count = 1600  # desired total images

original_images = [f for f in os.listdir(healthy_folder) if f.lower().endswith(('.jpg', '.png'))]

while len(original_images) + count < target_count:
    for img_name in original_images:
        img_path = os.path.join(healthy_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        augmented_img = augment(img)
        save_path = os.path.join(augmented_folder, f"aug_{count}_{img_name}")
        augmented_img.save(save_path)
        count += 1
        if len(original_images) + count >= target_count:
            break

print(f"✅ Augmentation done! Total augmented images added: {count}")
