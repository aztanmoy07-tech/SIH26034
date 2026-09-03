import os
import cv2
import numpy as np
import glob
import random
from pathlib import Path

def get_random_background(width, height):
    """Generates a synthetic realistic background (e.g., table/shelf)."""
    # Create a base color (e.g., wood brown or dark shelf)
    colors = [
        (40, 50, 70),   # dark gray/blue
        (100, 150, 200),# light blueish
        (60, 80, 130),  # brownish wood (BGR)
        (200, 210, 220) # light countertop
    ]
    base_color = random.choice(colors)
    bg = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add noise / texture
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    bg = cv2.add(bg, noise)
    
    # Add a gradient shadow (simulating lighting)
    gradient = np.linspace(1, 0.5, height).reshape(height, 1, 1)
    bg = (bg * gradient).astype(np.uint8)
    return bg

def apply_perspective_warp(image, bboxes):
    """Applies a random perspective warp to simulate a photo taken from an angle."""
    h, w = image.shape[:2]
    
    # Random corner offsets (10-20% of image size)
    dx = int(w * random.uniform(0.1, 0.2))
    dy = int(h * random.uniform(0.1, 0.2))
    
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Randomly pick a warp type (tilt left, tilt right, pitch up, pitch down)
    warp_type = random.choice(['tilt_left', 'tilt_right', 'pitch_up', 'pitch_down'])
    
    if warp_type == 'tilt_left':
        pts2 = np.float32([[0, dy], [w, 0], [0, h-dy], [w, h]])
    elif warp_type == 'tilt_right':
        pts2 = np.float32([[0, 0], [w, dy], [0, h], [w, h-dy]])
    elif warp_type == 'pitch_up':
        pts2 = np.float32([[dx, 0], [w-dx, 0], [0, h], [w, h]])
    else: # pitch_down
        pts2 = np.float32([[0, 0], [w, 0], [dx, h], [w-dx, h]])

    # Calculate homography
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Apply warp to image
    warped_img = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Apply warp to bboxes
    new_bboxes = []
    for bbox in bboxes:
        cls_id, x_c, y_c, bw, bh = bbox
        
        # Convert YOLO to corner points
        x1 = (x_c - bw/2) * w
        y1 = (y_c - bh/2) * h
        x2 = (x_c + bw/2) * w
        y2 = (y_c + bh/2) * h
        
        corners = np.array([
            [[x1, y1]],
            [[x2, y1]],
            [[x2, y2]],
            [[x1, y2]]
        ], dtype=np.float32)
        
        # Warp corners
        warped_corners = cv2.perspectiveTransform(corners, matrix)
        
        # Get new bounding box
        new_x1 = np.min(warped_corners[:, 0, 0])
        new_y1 = np.min(warped_corners[:, 0, 1])
        new_x2 = np.max(warped_corners[:, 0, 0])
        new_y2 = np.max(warped_corners[:, 0, 1])
        
        # Clip to bounds
        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
        new_x2, new_y2 = min(w, new_x2), min(h, new_y2)
        
        # Back to YOLO format if valid
        if new_x2 > new_x1 and new_y2 > new_y1:
            new_xc = (new_x1 + new_x2) / 2.0 / w
            new_yc = (new_y1 + new_y2) / 2.0 / h
            new_w = (new_x2 - new_x1) / w
            new_h = (new_y2 - new_y1) / h
            new_bboxes.append((cls_id, new_xc, new_yc, new_w, new_h))
            
    return warped_img, new_bboxes, pts2

def add_glare(image):
    """Adds a realistic specular highlight (glare) simulating camera flash on plastic."""
    h, w = image.shape[:2]
    glare = np.zeros((h, w, 3), dtype=np.uint8)
    
    cx = random.randint(int(w*0.2), int(w*0.8))
    cy = random.randint(int(h*0.2), int(h*0.8))
    radius = random.randint(int(w*0.2), int(w*0.5))
    
    cv2.circle(glare, (cx, cy), radius, (255, 255, 255), -1)
    glare = cv2.GaussianBlur(glare, (99, 99), 0)
    
    # Alpha blend
    alpha = random.uniform(0.15, 0.4)
    return cv2.addWeighted(image, 1, glare, alpha, 0)

def generate_real_photos(input_dir, output_dir, sample_size=5000):
    print(f"[*] Upgrading dataset to 'Real Photo' augmentations...")
    img_dir = Path(input_dir) / "images" / "train"
    lbl_dir = Path(input_dir) / "labels" / "train"
    
    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    random.shuffle(images)
    images = images[:sample_size]
    
    print(f"[*] Processing {len(images)} images to simulate real physical packets...")
    
    success_count = 0
    for i, img_path in enumerate(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # Parse existing YOLO labels
        bboxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        
        # 1. Perspective Warp (simulate holding the packet at an angle)
        warped_img, warped_bboxes, pts = apply_perspective_warp(img, bboxes)
        
        # 2. Add realistic background
        bg = get_random_background(w, h)
        # Create mask from the warped packet
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(pts), 255)
        
        # Combine
        fg_masked = cv2.bitwise_and(warped_img, warped_img, mask=mask)
        bg_masked = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
        final_img = cv2.add(fg_masked, bg_masked)
        
        # 3. Add glare/flash reflection (simulating plastic packaging)
        if random.random() > 0.5:
            final_img = add_glare(final_img)
            
        # 4. Save new image and labels
        new_img_name = f"real_photo_{img_path.name}"
        new_img_path = img_dir / new_img_name
        new_lbl_path = lbl_dir / f"{Path(new_img_name).stem}.txt"
        
        cv2.imwrite(str(new_img_path), final_img)
        with open(new_lbl_path, "w") as f:
            for bbox in warped_bboxes:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                
        success_count += 1
        if success_count % 100 == 0:
            print(f"    - Generated {success_count} real-world packet photos...")
            
    print(f"[OK] Successfully added {success_count} real-world physical packet variations to the dataset.")

if __name__ == "__main__":
    generate_real_photos("dataset_yolo_full", "dataset_yolo_full", sample_size=10000)
