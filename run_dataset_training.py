"""
=============================================================================
MetriGuard — 80k Dataset Loader & YOLO11 Fast-Trainer
=============================================================================
1. Streams & extracts seed batch from C:/Users/ajtan/Downloads/archive.zip (81,103 images).
2. Sets up YOLO train/val folders and annotations.
3. Kicks off YOLO11 training.
=============================================================================
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from PIL import Image

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[!] Run: pip install ultralytics torch")
    sys.exit(1)


def setup_dataset_from_zip(zip_path: str, target_dir: str = "dataset_yolo", sample_count: int = 200):
    """
    Extracts a sample subset of images from archive.zip and structures them for YOLO.
    """
    print(f"[*] Reading archive zip: {zip_path}")
    target_path = Path(target_dir).resolve()
    
    train_img_dir = target_path / "images" / "train"
    val_img_dir = target_path / "images" / "val"
    train_lbl_dir = target_path / "labels" / "train"
    val_lbl_dir = target_path / "labels" / "val"
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX')]
        print(f"[*] Found {len(all_files)} total images in zip archive.")
        
        selected_files = all_files[:sample_count]
        split_idx = int(len(selected_files) * 0.8)
        
        train_files = selected_files[:split_idx]
        val_files = selected_files[split_idx:]
        
        print(f"[*] Extracting {len(train_files)} train images and {len(val_files)} val images...")
        
        for idx, f in enumerate(train_files):
            filename = f"train_sample_{idx}.png"
            dest_img = train_img_dir / filename
            dest_lbl = train_lbl_dir / f"train_sample_{idx}.txt"
            
            with z.open(f) as zf, open(dest_img, "wb") as out_f:
                shutil.copyfileobj(zf, out_f)
            
            # Synthetic label for package label detection (Class 0: commodity_package, centered bbox)
            with open(dest_lbl, "w") as lf:
                lf.write("0 0.5 0.5 0.8 0.8\n")

        for idx, f in enumerate(val_files):
            filename = f"val_sample_{idx}.png"
            dest_img = val_img_dir / filename
            dest_lbl = val_lbl_dir / f"val_sample_{idx}.txt"
            
            with z.open(f) as zf, open(dest_img, "wb") as out_f:
                shutil.copyfileobj(zf, out_f)
            
            with open(dest_lbl, "w") as lf:
                lf.write("0 0.5 0.5 0.8 0.8\n")

    # Generate data.yaml
    yaml_path = target_path / "data.yaml"
    yaml_content = f"""
path: {target_path.as_posix()}
train: images/train
val: images/val

names:
  0: commodity_package
  1: mrp_declaration
  2: net_quantity
  3: fssai_license
"""
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content.strip())
        
    print(f"[✓] Dataset structured successfully at: {target_path}")
    return str(yaml_path)


def start_training(yaml_path: str, epochs: int = 5, model_variant: str = "yolo11n.pt"):
    """
    Executes YOLO11 training.
    """
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting YOLO11 ({model_variant}) training for {epochs} epochs on device: {device.upper()}...")
    
    model = YOLO(model_variant)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=320,  # 320 for quick test iteration, 640 for full production
        batch=8,
        device=device,
        project="metriguard_yolo_runs",
        name="sih26034_yolo11",
        save=True,
        plots=True,
        exist_ok=True
    )
    print(f"[✓] YOLO11 Training run completed! Weights saved to {results.save_dir}")
    return results


if __name__ == "__main__":
    zip_location = r"C:\Users\ajtan\Downloads\archive.zip"
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    yaml_file = setup_dataset_from_zip(zip_location, sample_count=100)
    start_training(yaml_file, epochs=epochs)
