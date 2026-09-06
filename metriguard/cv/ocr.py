import cv2
import numpy as np
from typing import List, Dict, Optional

# Import domain schemas
from ..domain.schemas import Confidence, EvidenceCrop

_OCR_ENGINE = None

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
        except ImportError:
            _OCR_ENGINE = False
    return _OCR_ENGINE

class EvidenceExtractor:
    """Phase 4: OCR, Panel Classification, and Evidence Generation"""
    
    @classmethod
    def _preprocess_for_ocr(cls, img_np: np.ndarray) -> List[np.ndarray]:
        """Multi-pass image enhancement for OCR."""
        variants = []
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Variant 1: Original Grayscale
        variants.append(gray)
        
        # Variant 2: CLAHE + Sharpening (for dense text)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        variants.append(sharpened)
        
        return variants

    @classmethod
    def run_ocr(cls, img_np: np.ndarray, image_path: str = "temp.jpg") -> List[Dict]:
        """
        Runs multi-pass OCR and returns de-duplicated tokens mapped to evidence crops.
        """
        engine = get_ocr_engine()
        if not engine:
            return [] # Fallback for headless testing
            
        variants = cls._preprocess_for_ocr(img_np)
        all_results = []
        
        for variant in variants:
            result, _ = engine(variant)
            if result:
                all_results.extend(result)
                
        # Simple spatial de-duplication (naive for demo)
        unique_tokens = []
        seen_texts = set()
        
        for r in all_results:
            bbox = r[0] # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            text = r[1]
            conf = r[2]
            
            if text.lower() not in seen_texts:
                seen_texts.add(text.lower())
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                rect_bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                
                unique_tokens.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": rect_bbox,
                    "evidence_crop": EvidenceCrop(
                        image_path=image_path,
                        bbox=rect_bbox,
                        description=f"Extracted: {text}"
                    )
                })
                
        return unique_tokens
