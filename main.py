import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from glob import glob
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ==============================================
# CONFIGURATION
# ==============================================
TEAM_NAME = "ViksithBharat2047"

# Base directory (auto-detected)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Actual folder names on your system
REAL_DIR = os.path.join(DATA_DIR, "real_cifake_images-20251109T090423Z-1-001")
FAKE_DIR = os.path.join(DATA_DIR, "fake_cifake_images-20251109T090416Z-1-001")
TEST_DIR = os.path.join(DATA_DIR, "test-20251109T090525Z-1-001", "test")

REAL_JSON = os.path.join(DATA_DIR, "real_cifake_preds.json")
FAKE_JSON = os.path.join(DATA_DIR, "fake_cifake_preds.json")
TRAIN_JSON = os.path.join(DATA_DIR, "train_outputs.json")
OUTPUT_JSON = os.path.join(DATA_DIR, f"{TEAM_NAME}_prediction.json")

# ==============================================
# STEP 1: Merge JSON Files
# ==============================================
print("🔄 Merging JSONs...")
with open(REAL_JSON, 'r') as f:
    real_data = json.load(f)
with open(FAKE_JSON, 'r') as f:
    fake_data = json.load(f)

train_data = real_data + fake_data
with open(TRAIN_JSON, 'w') as f:
    json.dump(train_data, f, indent=2)
print(f"✅ Merged train_outputs.json saved with {len(train_data)} entries.")


# ==============================================
# STEP 2: Combine Real + Fake Images into /train
# ==============================================
train_dir = os.path.join(DATA_DIR, "train")
os.makedirs(train_dir, exist_ok=True)

print("📂 Combining real + fake images into /train...")
for src_dir in [REAL_DIR, FAKE_DIR]:
    for root, _, files in os.walk(src_dir):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(train_dir, f)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[WARN] Could not copy {f}: {e}")
print("✅ Training images copied successfully!")

# Detect file extension
sample_files = [f for f in os.listdir(train_dir) if Path(f).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
if not sample_files:
    raise RuntimeError("No images found in train folder.")
sample_ext = Path(sample_files[0]).suffix
print("🧩 Detected image extension:", sample_ext)

# ==============================================
# STEP 3: Match JSON to Training Images
# ==============================================
file_names, targets = [], []

for row in train_data:
    idx = str(row["index"])
    label = row["prediction"]

    # most of your filenames are plain numbers: 1.png, 2.png, ...
    # this loop tries a few variations just in case
    patterns = [
        f"{idx}{sample_ext}",
        f"{label}_{idx}{sample_ext}",
        f"{label}-{idx}{sample_ext}",
        f"{int(idx):04d}{sample_ext}",
    ]

    found = False
    for fname in patterns:
        if os.path.exists(os.path.join(train_dir, fname)):
            file_names.append(fname)
            targets.append(label)
            found = True
            break
    if not found:
        # optional: uncomment next line to debug any missing matches
        # print(f"[WARN] No match for index {idx} ({label})")
        pass

print(f"✅ Matched {len(file_names)} / {len(train_data)} images.")


# ==============================================
# STEP 4: Feature Extractor (EfficientNet)
# ==============================================
class Embedder(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.extractor = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.extractor.eval().to(self.device)
        self.tf = models.EfficientNet_B0_Weights.DEFAULT.transforms()

    @torch.no_grad()
    def embed_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {path}: {e}")
            return np.zeros(1280)  # EfficientNet-B0 embedding size
        x = self.tf(img).unsqueeze(0).to(self.device)
        return self.extractor(x).cpu().numpy()[0]

    def embed_batch(self, paths):
        return np.vstack([self.embed_image(p) for p in tqdm(paths, desc="Extracting features")])

# ==============================================
# STEP 5: Extract or Load Cached Embeddings
# ==============================================
features_path = os.path.join(DATA_DIR, "train_features.npy")
labels_path = os.path.join(DATA_DIR, "train_labels.npy")

le = LabelEncoder()  # Global label encoder

if os.path.exists(features_path) and os.path.exists(labels_path):
    print("📂 Loading cached features...")
    train_feats = np.load(features_path)
    y = np.load(labels_path)
    # ✅ Ensure encoder always fitted even when loading cached data
    le.fit(targets)
else:
    print("⚙️ Extracting image embeddings...")
    embedder = Embedder()
    train_paths = [os.path.join(train_dir, f) for f in file_names]

    if len(train_paths) == 0:
        raise RuntimeError("No training images found! Please verify your filename mapping pattern.")

    train_feats = embedder.embed_batch(train_paths)
    y = le.fit_transform(targets)

    np.save(features_path, train_feats)
    np.save(labels_path, y)
    print("✅ Cached features saved for reuse.")

# ==============================================
# STEP 6: Train Classifier
# ==============================================
print("\n🎯 Training classifier...")
X_tr, X_va, y_tr, y_va = train_test_split(train_feats, y, test_size=0.15, random_state=42)
clf = LogisticRegression(max_iter=2000, solver='lbfgs')
clf.fit(X_tr, y_tr)

acc = (clf.predict(X_va) == y_va).mean()
print(f"✅ Validation accuracy: {acc:.4f}")

# ==============================================
# STEP 7: Predict on Test Images
# ==============================================
print("\n🧪 Generating predictions on test set...")

test_images = sorted(glob(os.path.join(TEST_DIR, "**", "*.*"), recursive=True))
test_images = [p for p in test_images if Path(p).suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]

if len(test_images) == 0:
    raise RuntimeError(f"No test images found in {TEST_DIR}")

print(f"✅ Total test images found: {len(test_images)}")

embedder = Embedder()  # ✅ Now Python knows what Embedder is
test_feats = embedder.embed_batch(test_images)

preds = clf.predict(test_feats)
pred_labels = le.inverse_transform(preds)

# Save predictions
output = [{"image": Path(img).name, "prediction": label} for img, label in zip(test_images, pred_labels)]
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved predictions to {OUTPUT_JSON}")
print("Sample predictions:", output[:5])
