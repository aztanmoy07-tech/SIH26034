"""
=============================================================================
MetriGuard — High-Accuracy CV & OCR Pipeline v2.0
Multi-pass image enhancement + RapidOCR for maximum text extraction from
real-world product labels (noisy, glary, small text, curved surfaces).
=============================================================================
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, List, Dict, Any, Optional

_OCR_ENGINE = None

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
            print("[*] RapidOCR initialized successfully.")
        except BaseException as e:
            print(f"[!] RapidOCR unavailable: {e}")
            _OCR_ENGINE = False
    return _OCR_ENGINE


def _preprocess_for_ocr(img_np: np.ndarray) -> List[np.ndarray]:
    """
    Returns multiple enhanced variants of the image for multi-pass OCR.
    Each variant is tuned to recover different types of text
    (dark on light, light on dark, small text, glary labels etc.)
    """
    variants = []
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- Variant 1: CLAHE + Sharpening (best for dense small text) ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    variants.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))

    # --- Variant 2: Adaptive Threshold Binarization (best for tables & structured text) ---
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB))

    # --- Variant 3: Upscaled original (best for low-res inputs) ---
    h, w = img_np.shape[:2]
    if max(h, w) < 1400:
        scale = 1400.0 / max(h, w)
        upscaled = cv2.resize(img_np, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)
    else:
        variants.append(img_np)

    # --- Variant 4: Inverted (catches light-on-dark text sections) ---
    inverted = cv2.bitwise_not(gray)
    clahe_inv = clahe.apply(inverted)
    variants.append(cv2.cvtColor(clahe_inv, cv2.COLOR_GRAY2RGB))

    return variants


def _merge_tokens(all_token_sets: List[List[Dict]]) -> List[Dict]:
    """
    Merges OCR results from multiple image variants.
    Deduplicates based on spatial proximity and keeps highest-confidence reading.
    """
    if not all_token_sets:
        return []

    # Use first set as base, add new tokens from subsequent sets if they don't overlap
    merged = list(all_token_sets[0])

    for token_set in all_token_sets[1:]:
        for candidate in token_set:
            # Check if this token overlaps with any existing merged token
            cx1, cy1, cx2, cy2 = candidate["bbox"]
            is_duplicate = False
            for existing in merged:
                ex1, ey1, ex2, ey2 = existing["bbox"]
                # Calculate intersection over union
                ix1, iy1 = max(cx1, ex1), max(cy1, ey1)
                ix2, iy2 = min(cx2, ex2), min(cy2, ey2)
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area_c = max(1, (cx2 - cx1) * (cy2 - cy1))
                    area_e = max(1, (ex2 - ex1) * (ey2 - ey1))
                    iou = intersection / min(area_c, area_e)
                    if iou > 0.4:
                        # Keep higher confidence reading
                        if candidate["confidence"] > existing["confidence"]:
                            existing["text"] = candidate["text"]
                            existing["confidence"] = candidate["confidence"]
                        is_duplicate = True
                        break
            if not is_duplicate:
                # Only add if confidence is reasonable
                if candidate["confidence"] >= 0.4:
                    merged.append(candidate)

    # Sort by vertical position (top to bottom, then left to right)
    merged.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    return merged


class ImagePreprocessor:

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """
        Smart background removal using contour detection.
        Preserves label text regions — does NOT blur or distort text.
        """
        try:
            img_np = np.array(image.convert("RGB"))
            h, w = img_np.shape[:2]
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > (w * h * 0.10):
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    isolated = np.full(img_np.shape, 255, dtype=np.uint8)
                    isolated[mask == 255] = img_np[mask == 255]
                    return Image.fromarray(isolated)

            margin_x, margin_y = max(1, int(w * 0.04)), max(1, int(h * 0.04))
            rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
            mask = np.zeros((h, w), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img_np, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            isolated = np.full(img_np.shape, 255, dtype=np.uint8)
            isolated[mask2 == 1] = img_np[mask2 == 1]
            return Image.fromarray(isolated)
        except BaseException:
            return image

    @staticmethod
    def assess_quality(image: Image.Image) -> Dict[str, Any]:
        img_np = np.array(image.convert("L"))
        h, w = img_np.shape
        laplacian_var = float(cv2.Laplacian(img_np, cv2.CV_64F).var())
        glare_ratio = float(np.sum(img_np > 250)) / (h * w)
        is_low_res = (w < 400 or h < 400)
        return {
            "is_blurry": laplacian_var < 50.0,
            "blur_score": round(laplacian_var, 1),
            "has_glare": glare_ratio > 0.20,
            "glare_ratio": round(glare_ratio * 100, 1),
            "is_low_res": is_low_res,
            "resolution": f"{w}x{h} px",
            "quality_grade": "GOOD" if laplacian_var >= 50.0 and glare_ratio <= 0.20 and not is_low_res else "MANUAL_REVIEW_RECOMMENDED"
        }

    @staticmethod
    def detect_aruco_calibration(image: Image.Image) -> Tuple[float, Optional[List[int]]]:
        try:
            img_np = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(corners) > 0:
                pts = corners[0][0]
                side_px = np.linalg.norm(pts[0] - pts[1])
                return 50.0 / float(side_px), [int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())]
        except BaseException:
            pass
        return 0.1, None


class PackageExtractor:

    @staticmethod
    def run_ocr(image: Image.Image) -> List[Dict[str, Any]]:
        """
        Multi-pass OCR with image enhancement.
        Runs RapidOCR on multiple preprocessed variants of the image
        and merges the results to maximize text extraction accuracy.
        """
        engine = get_ocr_engine()
        if not engine:
            print("[!] OCR engine not available. Returning empty token list.")
            return []

        img_np = np.array(image.convert("RGB"))
        variants = _preprocess_for_ocr(img_np)

        all_token_sets = []
        for i, variant in enumerate(variants):
            try:
                result, _ = engine(variant)
                token_set = []
                if result:
                    for item in result:
                        pts, text, conf = item
                        pts = np.array(pts, dtype=np.int32)
                        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
                        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
                        h_px = max(1, y_max - y_min)
                        w_px = max(1, x_max - x_min)
                        text_clean = text.strip()
                        if text_clean and len(text_clean) >= 1 and float(conf) >= 0.40:
                            token_set.append({
                                "text": text_clean,
                                "confidence": round(float(conf), 3),
                                "bbox": [x_min, y_min, x_max, y_max],
                                "bbox_height_px": h_px,
                                "bbox_width_px": w_px,
                                "ocr_pass": i + 1
                            })
                if token_set:
                    all_token_sets.append(token_set)
                    print(f"[OCR] Pass {i+1}: extracted {len(token_set)} tokens")
            except BaseException as e:
                print(f"[OCR] Pass {i+1} error: {e}")

        if not all_token_sets:
            print("[!] All OCR passes failed or returned no text.")
            return []

        merged = _merge_tokens(all_token_sets)
        print(f"[OCR] Merged result: {len(merged)} unique tokens from {len(all_token_sets)} passes")
        return merged
