import sys
from metriguard.cv.parser import SpatialParser
from metriguard.rules.engine import CanonicalRulesEngine
from metriguard.domain.schemas import InspectionCase

def main():
    print("Running Smoke Test...")
    
    # 1. Split serving-size extraction works
    tokens = [
        {"text": "Serving", "bbox": [10, 10, 50, 20], "confidence": 0.99},
        {"text": "Size:", "bbox": [55, 10, 90, 20], "confidence": 0.98},
        {"text": "30", "bbox": [100, 10, 120, 20], "confidence": 0.99},
        {"text": "g", "bbox": [125, 10, 135, 20], "confidence": 0.97},
    ]
    rows = SpatialParser.group_by_rows(tokens)
    res = SpatialParser.extract_serving_size(rows)
    assert res['value'] == 30, "Smoke Test Failed: Split serving size"
    
    # 2. Values do not cross rows
    tokens_bad = [
        {"text": "Serving Size:", "bbox": [10, 10, 100, 20], "confidence": 0.95},
        {"text": "Energy", "bbox": [10, 30, 80, 40], "confidence": 0.99},
        {"text": "2000 kcal", "bbox": [100, 30, 180, 40], "confidence": 0.99},
        {"text": "30 g", "bbox": [10, 50, 40, 60], "confidence": 0.99}
    ]
    rows_bad = SpatialParser.group_by_rows(tokens_bad)
    res_bad = SpatialParser.extract_serving_size(rows_bad)
    assert res_bad is None, "Smoke Test Failed: Values crossed rows"
    
    # 3. Canonical engine runs
    engine = CanonicalRulesEngine()
    case = InspectionCase(user_id="test", role="test", organization="test", package_metadata={"raw_tokens": []})
    evaluated = engine.evaluate_case(case)
    assert evaluated is not None, "Smoke Test Failed: Canonical engine crash"
    
    print("Smoke Test Passed!")
    sys.exit(0)

if __name__ == "__main__":
    main()
