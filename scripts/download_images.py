# # scripts/download_images.py
# import os
# import csv
# import requests
# from PIL import Image
# from io import BytesIO

# DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
# IMAGES_DIR = os.path.join(DATA_DIR, 'images')
# os.makedirs(IMAGES_DIR, exist_ok=True)

# csv_path = os.path.join(DATA_DIR, 'products.csv')

# def download_image(url, out_path):
#     try:
#         resp = requests.get(url, timeout=10)
#         resp.raise_for_status()
#         img = Image.open(BytesIO(resp.content)).convert('RGB')
#         img.save(out_path, format='JPEG', quality=85)
#         return True
#     except Exception as e:
#         print(f"Failed {url}: {e}")
#         return False

# with open(csv_path, newline='', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         pid = row['id']
#         url = row['image_url']
#         out_file = os.path.join(IMAGES_DIR, f"{pid}.jpg")
#         if os.path.exists(out_file):
#             continue
#         ok = download_image(url, out_file)
#         if not ok:
#             print(f"Skipping {pid}")

# import os
# import csv
# import requests
# from PIL import Image
# from io import BytesIO

# # Define paths
# DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
# IMAGES_DIR = os.path.join(DATA_DIR, 'images')
# os.makedirs(IMAGES_DIR, exist_ok=True)

# csv_path = os.path.join(DATA_DIR, 'products.csv')

# def convert_unsplash_url(url):
#     """
#     Convert a normal Unsplash page URL to a direct raw image link.
#     Example:
#     https://unsplash.com/photos/nwOip8AOZz0  -->  
#     https://images.unsplash.com/photo-nwOip8AOZz0?auto=format&fit=crop&w=800
#     """
#     if "unsplash.com/photos/" in url:
#         photo_id = url.split("/")[-1].split("?")[0]
#         return f"https://images.unsplash.com/photo-{photo_id}?auto=format&fit=crop&w=800"
#     return url

# def download_image(url, out_path):
#     try:
#         resp = requests.get(url, timeout=10)
#         resp.raise_for_status()
#         img = Image.open(BytesIO(resp.content)).convert("RGB")
#         img.save(out_path, format="JPEG", quality=85)
#         return True
#     except Exception as e:
#         print(f"❌ Failed: {url} ({e})")
#         return False

# # Read CSV (supports both comma and tab)
# with open(csv_path, newline='', encoding='utf-8') as f:
#     reader = csv.reader(f, delimiter=',')
#     header = next(reader)

#     # Determine column indexes
#     id_idx = header.index('id')
#     url_idx = header.index('image_url')

#     for row in reader:
#         pid = row[id_idx].strip()
#         url = convert_unsplash_url(row[url_idx].strip())

#         out_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
#         if os.path.exists(out_path):
#             continue

#         if download_image(url, out_path):
#             print(f"✅ Downloaded: {pid}")
#         else:
#             print(f"⚠️ Skipped: {pid}")

import os
import csv
import requests
from PIL import Image
from io import BytesIO

# -----------------------------
# PATHS
# -----------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

csv_path = os.path.join(DATA_DIR, 'products.csv')

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def convert_unsplash_url(url):
    """
    Convert a normal Unsplash page URL to a direct raw image link.
    Example:
    https://unsplash.com/photos/nwOip8AOZz0  -->
    https://images.unsplash.com/photo-nwOip8AOZz0?auto=format&fit=crop&w=800
    """
    if "unsplash.com/photos/" in url:
        photo_id = url.split("/")[-1].split("?")[0]
        return f"https://images.unsplash.com/photo-{photo_id}?auto=format&fit=crop&w=800"
    return url

def download_and_resize_image(url, out_path, size=(224, 224)):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # Load and resize image
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)

        # Save as .jpg
        img.save(out_path, format="JPEG", quality=90)
        return True
    except Exception as e:
        print(f"❌ Failed: {url} ({e})")
        return False

# -----------------------------
# MAIN SCRIPT
# -----------------------------
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)

    id_idx = header.index('id')
    url_idx = header.index('image_url')

    for row in reader:
        pid = row[id_idx].strip()
        url = convert_unsplash_url(row[url_idx].strip())

        out_path = os.path.join(IMAGES_DIR, f"{pid}.jpg")
        if os.path.exists(out_path):
            print(f"⏩ Skipping (already exists): {pid}")
            continue

        if download_and_resize_image(url, out_path):
            print(f"✅ Downloaded and resized: {pid}")
        else:
            print(f"⚠️ Skipped: {pid}")
