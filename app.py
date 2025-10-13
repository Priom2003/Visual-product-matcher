import os
import streamlit as st
import faiss
import torch
import numpy as np
import joblib
from PIL import Image
from torchvision import models, transforms
from io import BytesIO
import requests
import base64

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="🖼️ Visual Product Matcher", layout="wide")
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "image_index.faiss")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# -----------------------------
# IMAGE TRANSFORM + MODEL
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

@torch.no_grad()
def extract_features(image, model):
    img_t = transform(image).unsqueeze(0)
    feat = model(img_t).squeeze().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat.astype('float32')

@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_PATH)
    metadata = joblib.load(METADATA_PATH)
    return index, metadata

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_similar(image, top_k=6):
    model = load_model()
    index, metadata = load_index()

    feat = extract_features(image, model).reshape(1, -1)
    distances, indices = index.search(feat, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx]
            results.append({
                "id": item["id"],
                "name": item["name"],
                "category": item["category"],
                "image_path": item["image_path"],
                "similarity": float(1 / (1 + dist))
            })
    return results

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# -----------------------------
# UI LAYOUT
# -----------------------------
st.title("🖼️ Visual Product Matcher")
st.markdown("Upload an image or paste an image URL to find visually similar products.")

# Load available categories
try:
    _, metadata = load_index()
    categories = sorted(list(set([item["category"] for item in metadata])))
except Exception:
    categories = []
    st.warning("⚠️ Could not load categories. Please ensure metadata.pkl exists.")

# Display available product categories (pill-style tags)
if categories:
    st.markdown("### 📦 Available Product Categories")
    cat_html = " ".join([
        f"<span style='display:inline-block; background-color:#e8f0fe; color:#1a73e8; "
        f"padding:6px 12px; margin:4px; border-radius:20px; font-size:14px;'>{c}</span>"
        for c in categories
    ])
    st.markdown(cat_html, unsafe_allow_html=True)

# Image input section
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    url_input = st.text_input("or Paste Image URL")

    query_image = None
    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")
    elif url_input:
        try:
            resp = requests.get(url_input)
            query_image = Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            st.error("⚠️ Failed to load image from URL")

    top_k = st.slider("Filter by Top Results", min_value=1, max_value=8, value=4, step=1)
    search_btn = st.button("🔍 Search Similar Products")

with col2:
    if query_image is not None:
        st.image(query_image, caption="Query Image", width=300)

# -----------------------------
# SEARCH RESULTS
# -----------------------------
if search_btn and query_image is not None:
    with st.spinner("Extracting features and searching..."):
        results = search_similar(query_image, top_k=top_k)

    st.subheader("🔍 Similar Products")

    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            with st.container():
                st.markdown(
                    "<div style='background-color:#f8f9fa; border-radius:15px; "
                    "padding:10px; margin-bottom:15px; text-align:center; "
                    "box-shadow:0 1px 3px rgba(0,0,0,0.1);'>",
                    unsafe_allow_html=True
                )
                st.image(r["image_path"], caption=None, width=224)
                st.markdown(f"**{r['name']}**  \n*{r['category']}*")
                st.caption(f"Similarity: {r['similarity']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
