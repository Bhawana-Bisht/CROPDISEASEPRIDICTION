import pandas as pd
import os
import shutil

# ===============================
# CONFIG
# ===============================
TRAIN_CSV = "train_dataset.csv"
VAL_CSV   = "val_dataset.csv"
TEST_CSV  = "test_dataset.csv"

OUTPUT_DIR = "dataset"   # yahan train/val/test folders banenge

# ===============================
# LOAD CSV FILES
# ===============================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ===============================
# AUTO-DETECT IMAGE PATH COLUMN
# ===============================
possible_path_cols = ["image_path", "filepath", "path", "image"]

image_col = None
for col in possible_path_cols:
    if col in train_df.columns:
        image_col = col
        break

if image_col is None:
    raise ValueError(
        f"❌ Image path column not found. Columns available: {train_df.columns.tolist()}"
    )

print(f"✅ Using image path column: {image_col}")

# ===============================
# FUNCTION TO COPY IMAGES
# ===============================
def copy_images(df, split_name):
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        img_path = str(row[image_col])
        label = str(row["label"])   # label int → string

        if not os.path.exists(img_path):
            print(f"❌ Missing image: {img_path}")
            missing += 1
            continue

        dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, os.path.basename(img_path))

        if not os.path.exists(dest_path):
            shutil.copy(img_path, dest_path)
            copied += 1

    print(f"✅ {split_name}: copied={copied}, missing={missing}")

# ===============================
# CREATE DATASET STRUCTURE
# ===============================
copy_images(train_df, "train")
copy_images(val_df, "val")
copy_images(test_df, "test")

print("\n🎉 DONE! Dataset folders created successfully.")
