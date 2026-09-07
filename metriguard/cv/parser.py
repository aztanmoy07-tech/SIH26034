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
    @staticmethod
    def get_center_x(bbox: List[int]) -> float:
        return (bbox[0] + bbox[2]) / 2.0

    @staticmethod
    def normalize_numeric_ocr(text: str) -> Tuple[float, str]:
        """Phase 2: Safe OCR Normalization (e.g. O.O g -> 0.0 g)"""
        txt = text.lower().replace(" ", "")
        txt = txt.replace('o', '0').replace('l', '1').replace(',', '.')
        
        match = re.search(r'(\d+(?:\.\d+)?)\s*([a-z%]+)?', txt)
        if match:
            return float(match.group(1)), match.group(2) if match.group(2) else ""
        return None, None

    @staticmethod
    def identify_columns(header_row: List[Dict]) -> List[Dict]:
        """Detects per 100g, per serving, %RDA columns from a header row"""
        columns = []
        for token in header_row:
            txt = token['text'].lower()
            if '100' in txt and ('g' in txt or 'ml' in txt):
                columns.append({"name": "per_100g", "label": "Per 100 g", "bbox": token['bbox'], "center_x": SpatialParser.get_center_x(token['bbox'])})
            elif 'serving' in txt or 'portion' in txt:
                columns.append({"name": "per_serving", "label": "Per serving", "bbox": token['bbox'], "center_x": SpatialParser.get_center_x(token['bbox'])})
            elif 'rda' in txt or 'daily' in txt or 'dv' in txt:
                columns.append({"name": "rda", "label": "% RDA", "bbox": token['bbox'], "center_x": SpatialParser.get_center_x(token['bbox'])})
        
        return sorted(columns, key=lambda c: c['center_x'])

    @staticmethod
    def extract_nutrient_row(row: List[Dict], columns: List[Dict], nutrient_keywords: List[str]) -> Dict:
        """Reads values left-to-right mapped to column geometry"""
        row_text = " ".join([t['text'].lower() for t in row])
        if not any(k in row_text for k in nutrient_keywords):
            return None
            
        # Determine the nutrient label found
        found_nutrient = next((k for k in nutrient_keywords if k in row_text), "unknown")
            
        result = {
            "nutrient": found_nutrient,
            "values": {},
            "evidence_tokens": row,
            "confidence": sum(t['confidence'] for t in row) / len(row)
        }
        
        if not columns:
            return result
            
        value_tokens = []
        for t in row:
            # Avoid the label itself
            if any(k in t['text'].lower() for k in nutrient_keywords):
                continue
            val, unit = SpatialParser.normalize_numeric_ocr(t['text'])
            if val is not None:
                value_tokens.append({"token": t, "val": val, "unit": unit, "center_x": SpatialParser.get_center_x(t['bbox'])})
                
        for v_tok in value_tokens:
            closest_col = min(columns, key=lambda c: abs(c['center_x'] - v_tok['center_x']))
            if abs(closest_col['center_x'] - v_tok['center_x']) < 150: # reasonable x-distance tolerance
                result['values'][closest_col['name']] = {
                    "value": v_tok['val'], 
                    "unit": v_tok['unit'],
                    "raw": v_tok['token']['text']
                }
                
        return result

    @staticmethod
    def extract_nutrition_table(rows: List[List[Dict]]) -> Dict:
        """Phase 3: Extracts a complete nutrition table safely."""
        columns = []
        parsed_rows = []
        
        # 1. Find Header Row
        for row in rows:
            row_text = " ".join([t['text'].lower() for t in row])
            if '100' in row_text or 'serving' in row_text or 'rda' in row_text:
                found_cols = SpatialParser.identify_columns(row)
                if found_cols:
                    columns = found_cols
                    break
                    
        # 2. Extract Nutrients
        nutrient_targets = [
            ("energy", ["energy", "calories", "kcal"]),
            ("protein", ["protein"]),
            ("carbohydrate", ["carbohydrate", "carbs"]),
            ("total_sugars", ["total sugar", "sugar"]),
            ("added_sugars", ["added sugar"]),
            ("total_fat", ["total fat", "fat"]),
            ("sodium", ["sodium"])
        ]
        
        for row in rows:
            for nut_key, keywords in nutrient_targets:
                res = SpatialParser.extract_nutrient_row(row, columns, keywords)
                if res:
                    res['nutrient'] = nut_key
                    parsed_rows.append(res)
                    break # Move to next OCR row once a nutrient is matched
                    
        return {
            "columns": columns,
            "rows": parsed_rows
        }
