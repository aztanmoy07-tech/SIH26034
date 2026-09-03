"""
MetriGuard — Web Data Downloader
Downloads additional Indian FMCG packaging images from Open Images V7 and
other public datasets to augment the 81k archive.zip training data.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image
import io


# ── Open Images V7 labels relevant to Indian FMCG packaging ──────────────
# These are Open Images class names that produce packaged goods images
OPEN_IMAGES_CLASSES = [
    "/m/01zl9v",   # Packaged goods
    "/m/0bt_c3",   # Snack food
    "/m/02y6n",    # Biscuit
    "/m/0dftk",    # Candy
    "/m/01c648",   # Noodle
    "/m/05z55",    # Juice
    "/m/014j1m",   # Food
    "/m/025dyy",   # Tin can
    "/m/0fm3zh",   # Cookie
    "/m/04c0y",    # Chocolate
    "/m/020lf",    # Milk
    "/m/0fj52s",   # Sauce bottle
    "/m/033cnk",   # Salt
]

OIDV7_BBOX_URL = "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
OIDV7_IMG_INFO = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"

# Kaggle dataset URLs (if logged in via KAGGLE_API_KEY)
KAGGLE_DATASETS = [
    "vbookshelf/synthetic-packaged-goods-labels",
    "datasets/indian-food-label",
]


def download_open_images_subset(out_dir: str, max_images: int = 5000):
    """
    Downloads packaged goods images from Open Images V7 using the OID toolkit
    or simple URL-based download.
    """
    target = Path(out_dir) / "web_data"
    target.mkdir(parents=True, exist_ok=True)

    print("[*] Fetching Open Images V7 packaged goods image list...")

    # Use the simplified Open Images downloader approach
    # Download image CSV from validation set (smaller, faster)
    val_csv_url = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
    val_csv_path = target / "val_bbox.csv"

    if not val_csv_path.exists():
        print("[*] Downloading Open Images validation bbox annotations (~90MB)...")
        try:
            urllib.request.urlretrieve(val_csv_url, str(val_csv_path))
            print("[OK] Downloaded validation bbox CSV")
        except Exception as e:
            print(f"[!] Could not download Open Images CSV: {e}")
            return use_fallback_downloads(out_dir, max_images)

    # Parse and filter for packaged goods
    image_ids = set()
    print("[*] Scanning annotations for packaged goods labels...")
    with open(val_csv_path, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                label = parts[2]
                if label in OPEN_IMAGES_CLASSES:
                    image_ids.add(parts[0])
                    if len(image_ids) >= max_images:
                        break

    print(f"[*] Found {len(image_ids)} relevant images. Downloading...")

    # Download image URLs from val image list
    val_imgs_url = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
    val_imgs_path = target / "val_images.csv"
    if not val_imgs_path.exists():
        urllib.request.urlretrieve(val_imgs_url, str(val_imgs_path))

    downloaded = 0
    with open(val_imgs_path, encoding='utf-8') as f:
        next(f)
        for line in f:
            if downloaded >= max_images:
                break
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            img_id, url = parts[0], parts[1]
            if img_id not in image_ids:
                continue
            try:
                out_path = target / f"oi_{img_id}.jpg"
                if not out_path.exists():
                    urllib.request.urlretrieve(url, str(out_path))
                    # Quick validation
                    with Image.open(out_path) as img:
                        if img.width < 100 or img.height < 100:
                            os.remove(str(out_path))
                            continue
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"   {downloaded}/{max_images} downloaded...")
            except Exception:
                pass

    print(f"[OK] Downloaded {downloaded} images to {target}")
    return downloaded


def use_fallback_downloads(out_dir: str, max_images: int = 200):
    """
    Fallback: Downloads sample FMCG label images from various public URLs
    when Open Images access fails.
    """
    target = Path(out_dir) / "web_data_fallback"
    target.mkdir(parents=True, exist_ok=True)

    # Wikipedia Commons FMCG images
    SAMPLE_URLS = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Maggi_noodles.jpg/800px-Maggi_noodles.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Lay%27s_potato_chips.jpg/640px-Lay%27s_potato_chips.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Britannia_Good_Day.jpg/640px-Britannia_Good_Day.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Parle-G_biscuit.jpg/640px-Parle-G_biscuit.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Amul_Butter.jpg/640px-Amul_Butter.jpg",
    ]

    downloaded = 0
    for i, url in enumerate(SAMPLE_URLS):
        try:
            out_path = target / f"sample_{i:04d}.jpg"
            if not out_path.exists():
                req = urllib.request.Request(url, headers={'User-Agent': 'MetriGuard/1.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                with open(out_path, 'wb') as f:
                    f.write(data)
                downloaded += 1
                print(f"   Downloaded: {url.split('/')[-1]}")
        except Exception as e:
            print(f"   [!] Failed: {url}: {e}")

    print(f"[OK] Fallback: {downloaded} images downloaded to {target}")
    return downloaded


def download_from_roboflow(out_dir: str, api_key: str = None, max_images: int = 1000):
    """
    Downloads Indian packaging label datasets from Roboflow Universe (if API key provided).
    Dataset: indian-food-packaging-labels, packaged-goods-detection
    """
    if not api_key:
        print("[!] No Roboflow API key. Set ROBOFLOW_API_KEY env var.")
        return

    import urllib.request, json
    workspace = "packaging-compliance"
    datasets = [
        f"https://api.roboflow.com/{workspace}/packaged-goods/1/yolov8?api_key={api_key}",
    ]

    for dataset_url in datasets:
        try:
            with urllib.request.urlopen(dataset_url) as r:
                info = json.load(r)
            print(f"[OK] Roboflow dataset info: {info}")
        except Exception as e:
            print(f"[!] Roboflow failed: {e}")


def move_to_training_dataset(web_data_dir: str, training_dir: str):
    """Moves downloaded web images into the YOLO training dataset."""
    from PIL import Image
    import shutil

    web_dir = Path(web_data_dir)
    train_img = Path(training_dir) / "images" / "train"
    train_lbl = Path(training_dir) / "labels" / "train"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_file in web_dir.rglob("*.jpg"):
        try:
            dst = train_img / f"web_{img_file.stem}.jpg"
            lbl_dst = train_lbl / f"web_{img_file.stem}.txt"
            shutil.copy2(str(img_file), str(dst))
            # Whole-image label (class 0 = principal_display_panel)
            with open(str(lbl_dst), 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0\n")
            count += 1
        except Exception:
            pass

    print(f"[OK] Moved {count} web images to training dataset at {training_dir}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="dataset_yolo_full")
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--roboflow-key", default=None)
    args = parser.parse_args()

    print(f"[*] MetriGuard Web Data Downloader — target: {args.max_images} images")

    n = download_open_images_subset(args.out_dir, args.max_images)
    if n < 10:
        n = use_fallback_downloads(args.out_dir)

    if args.roboflow_key:
        download_from_roboflow(args.out_dir, args.roboflow_key, args.max_images)

    if n > 0:
        moved = move_to_training_dataset(
            os.path.join(args.out_dir, "web_data"),
            args.out_dir
        )
        print(f"[*] Added {moved} web images to training dataset.")
    print("[DONE]")
