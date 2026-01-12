import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
FEATURE_CACHE = "data/resnet18_features.npz"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

DATA_DIR = "data"   # train/val/test folders inside this

CLASSES = ["Bacterialblight", "Brownspot", "Leafsmut", "HealthyLeaf"]

# -----------------------------------------------------------
# SAFE IMAGE READER
# -----------------------------------------------------------
def safe_read_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))

# -----------------------------------------------------------
# BUILD CSV FROM TRAIN/VAL/TEST FOLDERS
# -----------------------------------------------------------
def build_dataframe():
    rows = []
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            folder = os.path.join(DATA_DIR, split, cls)
            if not os.path.exists(folder):
                continue

            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({
                        "image_path": os.path.join(folder, f),
                        "label": cls,
                        "split": split
                    })

    df = pd.DataFrame(rows)
    df.to_csv("dataset_labels_generated.csv", index=False)
    print("CSV saved:", "dataset_labels_generated.csv")
    return df


df = build_dataframe()
print(df.head())

# -----------------------------------------------------------
# LABEL ENCODING
# -----------------------------------------------------------
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

print("Labels:", le.classes_)

# -----------------------------------------------------------
# FEATURE EXTRACTION USING RESNET18
# -----------------------------------------------------------
def extract_features(df, cache_path=FEATURE_CACHE):

    if os.path.exists(cache_path):
        print("🔁 Loading cached features...")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["paths"]

    print("⚡ Extracting features using ResNet18…")

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1])
    extractor.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    paths = df["image_path"].tolist()
    feats, labs = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE)):
            batch = paths[i:i + BATCH_SIZE]
            imgs = torch.stack([transform(safe_read_image(p)) for p in batch]).to(DEVICE)
            f = extractor(imgs).view(imgs.size(0), -1).cpu().numpy()

            feats.append(f)
            labs.extend(df["label_encoded"].iloc[i:i + len(batch)].tolist())

    feats = np.vstack(feats)
    labs = np.array(labs)

    np.savez_compressed(cache_path, features=feats, labels=labs, paths=np.array(paths))

    return feats, labs, np.array(paths)


features, labels, paths = extract_features(df)

# -----------------------------------------------------------
# SPLITTING EXACTLY BY TRAIN/VAL/TEST YOU CREATED
# -----------------------------------------------------------
train_idx = df[df["split"] == "train"].index
val_idx = df[df["split"] == "val"].index
test_idx = df[df["split"] == "test"].index

X_train, y_train, p_train = features[train_idx], labels[train_idx], paths[train_idx]
X_val, y_val, p_val = features[val_idx], labels[val_idx], paths[val_idx]
X_test, y_test, p_test = features[test_idx], labels[test_idx], paths[test_idx]

print(f"Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

# -----------------------------------------------------------
# ML MODELS
# -----------------------------------------------------------
ml_models = {
    "SVM": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))]),
    "RandomForest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    ),
    "KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7))]),
    "NaiveBayes": Pipeline([("scaler", StandardScaler()), ("nb", GaussianNB())]),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
}

ml_results = {}

print("\n========================")
print("     TRAINING ML MODELS ")
print("========================")

for name, model in ml_models.items():
    print("Training:", name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    ml_results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0)
    }

# -----------------------------------------------------------
# CNN TRAINING (MOBILENET-V2)
# -----------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = safe_read_image(self.paths[idx])
        return self.transform(img), self.labels[idx]

    def __len__(self):
        return len(self.paths)


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_loader = DataLoader(ImageDataset(p_train, y_train, img_transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageDataset(p_val, y_val, img_transform), batch_size=BATCH_SIZE)
test_loader = DataLoader(ImageDataset(p_test, y_test, img_transform), batch_size=BATCH_SIZE)

cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
cnn.classifier[1] = nn.Linear(cnn.classifier[1].in_features, len(le.classes_))
cnn.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)


def train_cnn(model, train_loader, val_loader, epochs=3):
    for ep in range(epochs):
        print(f"Epoch {ep + 1}/{epochs}")
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
    return model


cnn = train_cnn(cnn, train_loader, val_loader, epochs=3)


def cnn_eval(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labs.extend(yb.numpy())

    return {
        "accuracy": accuracy_score(labs, preds),
        "precision": precision_score(labs, preds, average="weighted"),
        "recall": recall_score(labs, preds, average="weighted"),
        "f1": f1_score(labs, preds, average="weighted")
    }


dl_results = {"CNN": cnn_eval(cnn, test_loader)}

# -----------------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------------
rows = []
for k, v in ml_results.items():
    rows.append({"model": k, "type": "ML", **v})

for k, v in dl_results.items():
    rows.append({"model": k, "type": "DL", **v})

results_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
results_df.to_csv("results_compare_models.csv", index=False)

print("\n========================")
print("✔ RESULTS SAVED → results_compare_models.csv")
print("========================")
print(results_df)
