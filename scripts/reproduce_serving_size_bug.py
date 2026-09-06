import re

def parse_serving_size_flat(tokens):
    """The vulnerable flat text join approach"""
    all_text = " ".join([t['text'].lower() for t in tokens])
    
    # Naive search for serving size
    match = re.search(r'serving\s*size.*?(\d+)\s*(g|ml|kg|l)', all_text)
    if match:
        return {"value": match.group(1), "unit": match.group(2)}
    return None

def run_reproduction():
    # Case 1: Serving size split across lines, but another large number is nearby
    # If joined flat: "serving size energy 2000 kcal 30 g" -> might borrow wrong number
    tokens_case_1 = [
        {"text": "Serving Size", "bbox": [10, 10, 100, 20]},
        {"text": "Energy", "bbox": [10, 30, 80, 40]},
        {"text": "2000 kcal", "bbox": [100, 30, 180, 40]},
        {"text": "30 g", "bbox": [120, 10, 160, 20]} # Geometrically same row as Serving Size
    ]
    
    print("Vulnerable Flat Extraction:")
    print("Result:", parse_serving_size_flat(tokens_case_1))
    print("EXPECTED: 30 g. ACTUAL (Due to flat order): Flat join might completely miss it or match wrongly based on list order.")

if __name__ == "__main__":
    run_reproduction()
