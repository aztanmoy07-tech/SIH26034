"""
=============================================================================
MetriGuard — YOLO11 Training & Active Learning Pipeline
=============================================================================
Supports:
1. Automated Dataset Prep from 'archive' directory.
2. Training latest YOLO11 (yolo11n-seg / yolo11m / yolo11-obb) for Package & Label detection.
3. Pseudo-labeling and high-confidence filtering for 80,000 images.
=============================================================================
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[!] Ultralytics or PyTorch not installed. Run: pip install ultralytics torch")
    sys.exit(1)


def prepare_dataset_yaml(archive_dir: str, output_yaml_path: str = "dataset.yaml"):
    """
    Scans the archive directory and generates the standard YOLO data.yaml configuration.
    """
    archive_path = Path(archive_dir).resolve()
    print(f"[*] Scanning archive dataset at: {archive_path}")

    classes = [
        "commodity_name",
        "manufacturer_details",
        "net_quantity",
        "mrp_declaration",
        "mfg_date",
        "consumer_care",
        "unit_sale_price",
        "fssai_logo",
        "veg_mark",
        "non_veg_mark",
        "barcode",
        "aruco_marker",
        "country_of_origin",
        "best_before",
        "qr_code_2023"
    ]

    yaml_content = f"""
path: {archive_path.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, cname in enumerate(classes):
        yaml_content += f"  {idx}: {cname}\n"

    with open(output_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print(f"[✓] Created dataset configuration at: {output_yaml_path}")
    return output_yaml_path


def train_yolo11(
    data_yaml: str = "dataset.yaml",
    model_size: str = "yolo11n-obb.pt",  # Oriented Bounding Boxes for tilted text/labels
    epochs: int = 100,                   # Increased epochs for better accuracy
    imgsz: int = 640,
    batch_size: int = 16,
    project_name: str = "metriguard_yolo11"
):
    """
    Initiates YOLO11 training on the specified dataset with hardware auto-detection.
    """
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing YOLO11 ({model_size}) on Device: {device.upper()}")
    if device != "cpu":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    model = YOLO(model_size)

    print(f"[*] Starting model training for {epochs} epochs at image size {imgsz}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_name,
        name="run_legal_metrology",
        save=True,
        plots=True,
        exist_ok=True,
        # Enhanced Augmentations for reading skewed/rotated product labels
        degrees=15.0,
        shear=10.0,
        perspective=0.001,
        hsv_s=0.5,
        hsv_v=0.4
    )
    print(f"[✓] Training Completed! Best weights saved in: {results.save_dir}")
    return results


def run_pseudo_labeling(model_path: str, unannotated_images_dir: str, confidence_threshold: float = 0.85):
    """
    Active Learning: Runs inference on remaining unannotated images from the 80k dataset,
    filtering for high confidence predictions (>= 0.85) to generate pseudo-labels.
    """
    print(f"[*] Loading model from {model_path} for active pseudo-labeling...")
    model = YOLO(model_path)
    
    image_files = glob.glob(os.path.join(unannotated_images_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(unannotated_images_dir, "**", "*.png"), recursive=True)
    
    print(f"[*] Found {len(image_files)} unannotated images to process.")
    accepted = 0
    
    for img_path in image_files:
        results = model.predict(source=img_path, conf=confidence_threshold, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                # Save label in YOLO format
                label_path = Path(img_path).with_suffix(".txt")
                r.save_txt(str(label_path))
                accepted += 1

    print(f"[✓] Pseudo-labeling complete! Generated {accepted} high-confidence annotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetriGuard YOLO11 Trainer")
    parser.add_argument("--archive", type=str, default="archive", help="Path to archive dataset folder")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO11 model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    # Look for archive directory in current or parent dirs if not explicitly found
    archive_dir = args.archive
    if not os.path.exists(archive_dir):
        alt_paths = ["../archive", "C:/Users/ajtan/archive", "C:/Users/ajtan/Downloads/archive"]
        for p in alt_paths:
            if os.path.exists(p):
                archive_dir = p
                break

    print(f"[*] Target Dataset Directory: {os.path.abspath(archive_dir)}")
    yaml_file = prepare_dataset_yaml(archive_dir)
    train_yolo11(data_yaml=yaml_file, model_size=args.model, epochs=args.epochs, batch_size=args.batch)
