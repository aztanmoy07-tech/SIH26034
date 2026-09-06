import pytest
from metriguard.cv.parser import SpatialParser

def test_regression_serving_size_split():
    """
    Phase 11: Regression Test #2
    'Serving', 'Size:', '30', and 'g' as separate tokens on the same row.
    """
    tokens = [
        {"text": "Serving", "bbox": [10, 10, 50, 20], "confidence": 0.99},
        {"text": "Size:", "bbox": [55, 10, 90, 20], "confidence": 0.98},
        {"text": "30", "bbox": [100, 10, 120, 20], "confidence": 0.99},
        {"text": "g", "bbox": [125, 10, 135, 20], "confidence": 0.97},
    ]
    
    rows = SpatialParser.group_by_rows(tokens)
    assert len(rows) == 1
    
    result = SpatialParser.extract_serving_size(rows)
    assert result is not None
    assert result['value'] == 30.0
    assert result['unit'] == 'g'
    assert result['bbox'] == [10, 10, 135, 20]

def test_regression_serving_size_nearby_large_number():
    """
    Phase 11: Regression Test #5 & #9
    A nearby row contains a larger number (e.g., 2000 kcal).
    The system must NOT borrow the value from the next row.
    """
    tokens = [
        # Row 1: Serving Size (Missing the actual size due to blur or cutoff)
        {"text": "Serving Size:", "bbox": [10, 10, 100, 20], "confidence": 0.95},
        
        # Row 2: Energy row directly below it
        {"text": "Energy", "bbox": [10, 30, 80, 40], "confidence": 0.99},
        {"text": "2000 kcal", "bbox": [100, 30, 180, 40], "confidence": 0.99},
        
        # Row 3: Random other row
        {"text": "30 g", "bbox": [10, 50, 40, 60], "confidence": 0.99}
    ]
    
    rows = SpatialParser.group_by_rows(tokens, y_tolerance=10)
    assert len(rows) == 3
    
    result = SpatialParser.extract_serving_size(rows)
    # Because '30 g' is on a DIFFERENT row (y=50..60 vs y=10..20), it must fail to find it.
    assert result is None
