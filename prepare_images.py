import os
import pandas as pd

# Base data folder
base_dir = os.path.join(os.getcwd(), "data")

# Dataset folder name (update if needed)
dataset_dir = os.path.join(base_dir, "rice leaf diseases dataset")

# Check dataset exists
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"❌ Dataset folder not found at {dataset_dir}")

print("✅ Dataset folder found!")

# Collect image paths and labels
data = []
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append([os.path.join(category_path, img_file), category])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["image_path", "label"])

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV file for reference
csv_path = os.path.join(base_dir, "dataset_labels.csv")
df.to_csv(csv_path, index=False)

print(f"📊 Dataset prepared successfully! Total images: {len(df)}")
print(f"📝 Labels saved to: {csv_path}")

# Show first few rows
print(df.head())
