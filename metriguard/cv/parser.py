from typing import List, Dict, Any, Tuple
import re

class ExtractedField:
    def __init__(self, field_name: str, raw_ocr_text: str, normalized_value: float, unit: str, 
                 confidence: float, bbox: List[int], manual_override: bool = False):
        self.field_name = field_name
        self.raw_ocr_text = raw_ocr_text
        self.normalized_value = normalized_value
        self.unit = unit
        self.confidence = confidence
        self.bbox = bbox
        self.manual_override = manual_override

class SpatialParser:
    """Phase 2 & 3: Safe OCR Normalization & Spatial Row Extraction"""
    
    @staticmethod
    def get_center_y(bbox: List[int]) -> float:
        """Returns the vertical center of a bounding box [x1, y1, x2, y2]."""
        return (bbox[1] + bbox[3]) / 2.0

    @staticmethod
    def group_by_rows(tokens: List[Dict], y_tolerance: int = 15) -> List[List[Dict]]:
        """
        Groups OCR tokens into rows based on vertical center proximity.
        tokens: [{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}]
        """
        if not tokens:
            return []
            
        # Sort tokens by Y-center
        sorted_tokens = sorted(tokens, key=lambda t: SpatialParser.get_center_y(t['bbox']))
        
        rows = []
        current_row = [sorted_tokens[0]]
        current_y_center = SpatialParser.get_center_y(sorted_tokens[0]['bbox'])
        
        for token in sorted_tokens[1:]:
            y_center = SpatialParser.get_center_y(token['bbox'])
            if abs(y_center - current_y_center) <= y_tolerance:
                current_row.append(token)
                # Update running average of row's center Y
                current_y_center = sum(SpatialParser.get_center_y(t['bbox']) for t in current_row) / len(current_row)
            else:
                # Sort the completed row by X coordinate (left to right)
                current_row.sort(key=lambda t: t['bbox'][0])
                rows.append(current_row)
                current_row = [token]
                current_y_center = y_center
                
        if current_row:
            current_row.sort(key=lambda t: t['bbox'][0])
            rows.append(current_row)
            
        return rows

    @staticmethod
    def extract_serving_size(rows: List[List[Dict]]) -> Dict:
        """
        Phase 3: Safely extracts Serving Size strictly from within the same spatial row.
        """
        serving_keywords = ['serving size', 'serving', 'portion size']
        
        for row in rows:
            row_text = " ".join([t['text'].lower() for t in row])
            if any(k in row_text for k in serving_keywords):
                # Search strictly within this row's tokens
                match = re.search(r'(\d+(?:\.\d+)?)\s*(g|kg|ml|l)\b', row_text)
                if match:
                    # Calculate bounding box encompassing the row
                    x1 = min(t['bbox'][0] for t in row)
                    y1 = min(t['bbox'][1] for t in row)
                    x2 = max(t['bbox'][2] for t in row)
                    y2 = max(t['bbox'][3] for t in row)
                    
                    return {
                        "value": float(match.group(1)),
                        "unit": match.group(2),
                        "raw": row_text,
                        "confidence": sum(t['confidence'] for t in row) / len(row),
                        "bbox": [x1, y1, x2, y2]
                    }
        return None
