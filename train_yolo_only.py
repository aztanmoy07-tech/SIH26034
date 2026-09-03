from ultralytics import YOLO
import torch
import os
from pathlib import Path

def get_counts():
    target = Path("dataset_yolo_full")
    train_imgs = list((target / "images" / "train").glob("*.png")) + list((target / "images" / "train").glob("*.jpg"))
    val_imgs = list((target / "images" / "val").glob("*.png")) + list((target / "images" / "val").glob("*.jpg"))
    return len(train_imgs), len(val_imgs)

if __name__ == "__main__":
    print("====================================================================")
    print("MetriGuard — Massive 2.8 Lakh Dataset YOLO11m Training")
    print("====================================================================")
    
    yaml_path = Path("dataset_yolo_full/metriguard_full.yaml")
    if not yaml_path.exists():
        # Recreate YAML just in case
        yaml_content = f"""
path: {yaml_path.parent.resolve().as_posix()}
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
        yaml_path.write_text(yaml_content.strip(), encoding='utf-8')

    try:
        t, v = get_counts()
        print(f"[*] Dataset size verified: {t} Training images, {v} Validation images")
        print(f"[*] Total massive dataset: {t+v} images ready for YOLO.")
    except:
        pass

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] YOLO11 Training with device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] WARNING: Running on CPU. Training will take an extremely long time on 2.8 Lakh images.")
        print("[!] Use the 'start_gpu_training.bat' file provided to use the Anaconda GPU environment.")

    # Use the pre-trained weights from the previous run to avoid download issues
    prev_weights = r"C:\Users\ajtan\runs\detect\metriguard_yolo_runs\sih26034_yolo11\weights\best.pt"
    if os.path.exists(prev_weights):
        print(f"[*] Resuming from previous run weights: {prev_weights}")
        model = YOLO(prev_weights)
    else:
        # Try to use yolo11n.pt if yolo11m.pt fails to download
        model = YOLO("yolo11n.pt") 

    run_dir = r"C:\Users\ajtan\runs\detect\metriguard_yolo_runs_full"
    os.makedirs(run_dir, exist_ok=True)

    print("[*] Starting YOLO model training...")
    try:
        results = model.train(
            data=str(yaml_path.resolve()),
            epochs=50,
            imgsz=640,
            batch=16,
            device=device,
            project=run_dir,
            name="massive_280k_training",
            exist_ok=True,
            save=True,
            save_period=5,
            workers=0
        )
        print("\n[OK] Training Complete!")
        print(f"[*] Best weights saved at: {run_dir}/massive_280k_training/weights/best.pt")
    except KeyboardInterrupt:
        print("\n[!] Training paused by user.")
    except Exception as e:
        print(f"\n[!] Training Error: {e}")
