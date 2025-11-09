🧠 Deepfake Detection Hackathon – Synergy’25
Team: ViksithBharat2047

Developed by: Aakash Jangeeti

📘 Project Overview

This project was built as part of Synergy’25 (IIIT Bangalore) for the Deepfake ML Model Hackathon.
The goal of the challenge is to design a predictive model that can detect whether an image is real or fake by learning from a labeled dataset and then generating predictions for unseen test images.

The task involved:

Merging real and fake image datasets

Extracting deep image embeddings using EfficientNet-B0

Training a lightweight classifier (Logistic Regression)

Generating a final JSON output predicting each test image as real or fake

🗂️ Dataset Description

The dataset provided consisted of:

Folder / File	Description
real_cifake_images-*	Folder containing real images
fake_cifake_images-*	Folder containing fake images
real_cifake_preds.json	JSON file with predictions for real images (provided by organizers)
fake_cifake_preds.json	JSON file with predictions for fake images (provided by organizers)
test-*	Folder containing unseen test images
train_outputs.json	Merged JSON used for training
ViksithBharat2047_prediction.json	Final output file (submission)
⚙️ Folder Structure
Synergy25/
│
├── fake_cifake_images-20251109T090416Z-1-001/
├── real_cifake_images-20251109T090423Z-1-001/
├── test-20251109T090525Z-1-001/
│   └── test/
│
├── real_cifake_preds.json
├── fake_cifake_preds.json
├── train_outputs.json
├── main.py
├── ViksithBharat2047_prediction.json
└── README.md

💻 How the Model Works
1. Data Preparation

The script merges both real_cifake_preds.json and fake_cifake_preds.json into a single JSON file (train_outputs.json).

It then consolidates real and fake images into a single /train directory.

2. Feature Extraction

A pre-trained EfficientNet-B0 model (from PyTorch) is used as a frozen feature extractor.

Each image is converted into a 1280-dimensional embedding vector.

3. Model Training

The extracted embeddings are used to train a Logistic Regression classifier.

The model predicts binary labels (real = 0, fake = 1).

A 15% validation split is used to measure accuracy.

4. Prediction on Test Data

The same feature extractor embeds all test images.

The trained classifier predicts their labels.

The predictions are saved in a final JSON file:

[
  {"image": "1.png", "prediction": "real"},
  {"image": "2.png", "prediction": "fake"},
  ...
]

🧩 Key Components
Component	Description
EfficientNet-B0	Pretrained CNN backbone used for embedding images
scikit-learn LogisticRegression	Lightweight linear classifier
LabelEncoder	Converts textual labels (real, fake) into numeric form
TQDM	Progress bar for embedding operations
PyTorch	Used for deep feature extraction
NumPy / JSON / Pillow	For data handling and I/O
🧰 Installation & Setup
1️⃣ Clone or download the repository
git clone https://github.com/<your-github>/Synergy25.git
cd Synergy25

2️⃣ Install required dependencies
pip install torch torchvision scikit-learn pillow tqdm numpy

3️⃣ Place all provided folders and JSON files in the same directory
Synergy25/
  ├── fake_cifake_images-...
  ├── real_cifake_images-...
  ├── test-.../test
  ├── fake_cifake_preds.json
  ├── real_cifake_preds.json
  └── main.py

4️⃣ Run the main script
python main.py

🧾 Output

The script generates:
✅ train_outputs.json (merged JSON for training)
✅ train_features.npy and train_labels.npy (cached embeddings)
✅ ViksithBharat2047_prediction.json (final submission file)
