import os
import pandas as pd

# ------------------------------
# CSV path
# ------------------------------
csv_path = "data/dataset_labels.csv"

# If the CSV exists, load it; otherwise create an empty DataFrame
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["image_path", "label"])

# ------------------------------
# Folder paths for each class
# Update these if you have other folders too
# ------------------------------
dataset_root = "data/rice leaf diseases dataset"

classes = ["Bacterialblight", "Brownspot", "Leafsmut", "HealthyLeaf"]

all_rows = []

for cls in classes:
    cls_dir = os.path.join(dataset_root, cls)
    if not os.path.exists(cls_dir):
        print(f"⚠️ Warning: Folder not found: {cls_dir}")
        continue

    for f in os.listdir(cls_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_rows.append({
                "image_path": os.path.join(cls_dir, f),
                "label": cls
            })

# ------------------------------
# Create new DataFrame
# ------------------------------
df_new = pd.DataFrame(all_rows)

# Save CSV
df_new.to_csv(csv_path, index=False)

print("✅ CSV file created/updated successfully!")
print(f"📁 Total images: {len(df_new)}")
print(f"📌 Classes: {df_new['label'].unique()}")
