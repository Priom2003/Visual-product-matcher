import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
import faiss
import joblib

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "products.csv")
INDEX_PATH = os.path.join(DATA_DIR, "image_index.faiss")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# -----------------------------
# IMAGE PREPROCESSING PIPELINE
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------------
# FEATURE EXTRACTOR (ResNet50)
# -----------------------------
def get_feature_extractor():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier head
    model.eval()
    return model


@torch.no_grad()
def extract_features(image_path, model):
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0)
        feat = model(img_t).squeeze().numpy()
        feat = feat / np.linalg.norm(feat)  # normalize
        return feat
    except Exception as e:
        print(f"❌ Error on {image_path}: {e}")
        return None


# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index():
    print("🔍 Loading model...")
    model = get_feature_extractor()

    embeddings = []
    metadata = []

    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Extracting image features"):
            pid = row['id']
            name = row['name']
            category = row['category']

            # image path
            img_path = os.path.join(IMAGE_DIR, f"{pid}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGE_DIR, f"{pid}.jpg")
            if not os.path.exists(img_path):
                print(f"⚠️ Missing image for ID {pid}")
                continue

            feat = extract_features(img_path, model)
            if feat is not None:
                embeddings.append(feat)
                metadata.append({
                    "id": pid,
                    "name": name,
                    "category": category,
                    "image_path": img_path
                })

    embeddings = np.vstack(embeddings).astype('float32')

    # save embeddings and metadata
    np.save(EMBEDDINGS_PATH, embeddings)
    joblib.dump(metadata, METADATA_PATH)

    # build faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ Index built with {len(metadata)} items.")
    print(f"📂 Saved to: {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
