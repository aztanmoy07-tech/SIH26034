"""
=============================================================================
MetriGuard — Full Archive YOLO11 Training Pipeline
Trains YOLO11m on ALL 81,103 images from archive.zip with proper dataset splitting.
Detects which PANEL of a packaging label is being analyzed.
=============================================================================
"""

import os
import sys
import json
import time
import zipfile
import shutil
from pathlib import Path
from PIL import Image
import io

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[!] Run: pip install ultralytics")
    sys.exit(1)


CLASSES = [
    "principal_display_panel",   # 0 - Front/PDP with MRP, Brand, Net Qty
    "nutritional_table",         # 1 - Back panel nutritional info
    "manufacturer_details",      # 2 - Mfg/Packer block
    "mrp_declaration",           # 3 - MRP label area
    "net_quantity",              # 4 - Net Qty text area
    "fssai_logo",                # 5 - FSSAI license & logo
    "veg_nonveg_mark",           # 6 - Green/Brown symbol
    "consumer_care",             # 7 - Helpline/email
    "barcode",                   # 8 - Barcode / QR
    "ingredient_list",           # 9 - Ingredients text block
]

def extract_all_images(zip_path: str, target_dir: str):
    """Extracts all 81,103 images from archive.zip into train/val splits."""
    target = Path(target_dir)
    train_img = target / "images" / "train"
    val_img = target / "images" / "val"
    train_lbl = target / "labels" / "train"
    val_lbl = target / "labels" / "val"

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[*] Opening archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = sorted([
            f for f in z.namelist()
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and '__MACOSX' not in f
        ])

        total = len(all_files)
        split = int(total * 0.85)
        train_files = all_files[:split]
        val_files = all_files[split:]

        print(f"[*] Total: {total} | Train: {len(train_files)} | Val: {len(val_files)}")

        for partition_name, files, img_dir, lbl_dir in [
            ("train", train_files, train_img, train_lbl),
            ("val", val_files, val_img, val_lbl)
        ]:
            print(f"[*] Extracting {partition_name} images...")
            for idx, fpath in enumerate(files):
                ext = os.path.splitext(fpath)[1].lower()
                out_name = f"{partition_name}_{idx:06d}{ext}"
                img_out = img_dir / out_name
                lbl_out = lbl_dir / (out_name.replace(ext, ".txt"))

                try:
                    with z.open(fpath) as zf:
                        data = zf.read()
                    
                    # Quick resize to save space and normalize
                    with Image.open(io.BytesIO(data)) as img:
                        img = img.convert("RGB")
                        img.save(str(img_out), optimize=True)
                    
                    # Heuristic label: classify as PDP vs nutritional vs other
                    fname_lower = fpath.lower()
                    if "nutri" in fname_lower or "back" in fname_lower or "ingredient" in fname_lower:
                        class_id = 1  # nutritional_table
                    elif "mrp" in fname_lower or "front" in fname_lower or "main" in fname_lower:
                        class_id = 0  # principal_display_panel
                    elif "barcode" in fname_lower or "qr" in fname_lower:
                        class_id = 8  # barcode
                    else:
                        class_id = 0  # default to PDP

                    # Whole-image bbox
                    with open(str(lbl_out), "w") as lf:
                        lf.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                except Exception as e:
                    pass  # Skip corrupt images silently

                if idx % 5000 == 0:
                    print(f"   [{partition_name}] {idx}/{len(files)} extracted...")

    # Write data.yaml
    yaml_content = f"""
path: {target.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: principal_display_panel
  1: nutritional_table
  2: manufacturer_details
  3: mrp_declaration
  4: net_quantity
  5: fssai_logo
  6: veg_nonveg_mark
  7: consumer_care
  8: barcode
  9: ingredient_list
"""
    yaml_path = target / "metriguard_full.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    print(f"[OK] Dataset fully extracted. YAML: {yaml_path}")
    return str(yaml_path)


def start_full_training(yaml_path: str, epochs: int = 50):
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] YOLO11 Training with device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo11m.pt")  # Best model for accuracy/speed balance

    start = time.time()
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=4 if device == "cpu" else 16,
        device=device,
        project="metriguard_yolo_runs_full",
        name="full_81k_training",
        save=True,
        plots=True,
        exist_ok=True,
        workers=0,
        cache=True,  # Cache dataset in RAM for speed
        patience=15,  # Early stopping if mAP doesn't improve for 15 epochs
    )
    elapsed = time.time() - start
    print(f"[OK] Training finished in {elapsed/3600:.2f} hours!")
    print(f"[OK] Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=r"C:\Users\ajtan\Downloads\archive.zip")
    parser.add_argument("--outdir", default="dataset_yolo_full")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction if already done")
    args = parser.parse_args()

    if not args.skip_extract:
        yaml = extract_all_images(args.zip, args.outdir)
    else:
        yaml = str(Path(args.outdir) / "metriguard_full.yaml")

    start_full_training(yaml, args.epochs)
