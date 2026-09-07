import pytest
from metriguard.cv.parser import SpatialParser

def test_regression_nutrition_table_columns():
    """
    Phase 11: Regression Test #4
    A nutrition table with Per 100 g, Per serving, and % RDA.
    """
    tokens = [
        # Header row
        {"text": "Nutritional", "bbox": [10, 10, 50, 20], "confidence": 0.9},
        {"text": "Info", "bbox": [55, 10, 80, 20], "confidence": 0.9},
        {"text": "Per 100g", "bbox": [100, 10, 150, 20], "confidence": 0.9},
        {"text": "Per Serving", "bbox": [180, 10, 250, 20], "confidence": 0.9},
        {"text": "% RDA", "bbox": [280, 10, 320, 20], "confidence": 0.9},
        
        # Protein row
        {"text": "Protein", "bbox": [10, 30, 60, 40], "confidence": 0.9},
        {"text": "8.4 g", "bbox": [110, 30, 140, 40], "confidence": 0.9},
        {"text": "2.5 g", "bbox": [190, 30, 220, 40], "confidence": 0.9},
        {"text": "5 %", "bbox": [290, 30, 310, 40], "confidence": 0.9}
    ]
    
    rows = SpatialParser.group_by_rows(tokens)
    table = SpatialParser.extract_nutrition_table(rows)
    
    assert len(table['columns']) == 3
    assert table['columns'][0]['name'] == 'per_100g'
    assert table['columns'][1]['name'] == 'per_serving'
    assert table['columns'][2]['name'] == 'rda'
    
    assert len(table['rows']) == 1
    protein_row = table['rows'][0]
    assert protein_row['nutrient'] == 'protein'
    assert protein_row['values']['per_100g']['value'] == 8.4
    assert protein_row['values']['per_serving']['value'] == 2.5
    assert protein_row['values']['rda']['value'] == 5.0

def test_regression_normalization_typo():
    """
    Phase 11: Regression Test #6
    A table where O.O g should become a cautious numeric normalization.
    """
    tokens = [
        {"text": "Per 100g", "bbox": [100, 10, 150, 20], "confidence": 0.9},
        {"text": "Fat", "bbox": [10, 30, 60, 40], "confidence": 0.9},
        {"text": "O.Og", "bbox": [110, 30, 140, 40], "confidence": 0.7} # OCR read 0 as O
    ]
    
    rows = SpatialParser.group_by_rows(tokens)
    table = SpatialParser.extract_nutrition_table(rows)
    
    assert len(table['rows']) == 1
    fat_row = table['rows'][0]
    assert fat_row['values']['per_100g']['value'] == 0.0
    assert fat_row['values']['per_100g']['unit'] == 'g'

