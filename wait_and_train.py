import time
import os
import subprocess
from pathlib import Path

def count_images():
    target = Path("dataset_yolo_full")
    if not target.exists():
        return 0
    t = sum(1 for _ in (target / "images" / "train").glob("*.*"))
    v = sum(1 for _ in (target / "images" / "val").glob("*.*"))
    return t + v

if __name__ == "__main__":
    print("[*] Waiting for 200,000 synthetic images to finish generating...")
    target_count = 200000 
    
    while True:
        current = count_images()
        if current >= target_count:
            print(f"\n[*] Massive Dataset generation complete! (Found {current} images)")
            break
        print(f"   ... Waiting. Currently {current}/{target_count} images ready. (Checking again in 30s)")
        time.sleep(30)

    print("[*] Launching YOLO Training on CPU (Fallback)...")
    print("[!] NOTE: For GPU acceleration, please run start_gpu_training.bat instead!")
    
    subprocess.run(["python", "train_yolo_only.py"])
