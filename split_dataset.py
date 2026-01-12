import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# Load full dataset
# -------------------------------
FULL_DATASET_CSV = "full_dataset.csv"  # Make sure this file exists
df = pd.read_csv(FULL_DATASET_CSV)

print(f"Total images in dataset: {len(df)}")
print(f"Classes found: {df['label'].unique().tolist()}")

# -------------------------------
# Step 1: Split 70% training and 30% temp (validation + test)
# -------------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,           # 30% for val + test
    stratify=df['label'],    # Maintain class proportions
    random_state=42
)

# -------------------------------
# Step 2: Split temp into 50% validation, 50% test
# 30% * 50% = 15% each
# -------------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,           # Half of temp → 15% of total
    stratify=temp_df['label'],
    random_state=42
)

# -------------------------------
# Step 3: Save CSV files
# -------------------------------
train_df.to_csv("train_dataset.csv", index=False)
val_df.to_csv("val_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)

# -------------------------------
# Step 4: Print summary
# -------------------------------
print("✅ Dataset split completed!")
print(f"Train images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")
