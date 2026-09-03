from legal_rules import LegalMetrologyRulesEngine

maggi_tokens = [
    {'text': 'Nutritional Information', 'confidence': 0.99, 'bbox': [0,0,400,30], 'bbox_height_px': 30, 'bbox_width_px': 400},
    {'text': 'Servings per pack: 6, Serving Size: 30g', 'confidence': 0.98, 'bbox': [0,40,350,60], 'bbox_height_px': 20, 'bbox_width_px': 350},
    {'text': 'Energy (kcal) 446 112', 'confidence': 0.97, 'bbox': [0,70,350,90], 'bbox_height_px': 20, 'bbox_width_px': 350},
    {'text': 'Total Fat (g) 17.9 4.5', 'confidence': 0.96, 'bbox': [0,100,300,120], 'bbox_height_px': 20, 'bbox_width_px': 300},
    {'text': 'Saturated Fat (g) 8.1 2.0', 'confidence': 0.95, 'bbox': [0,120,300,140], 'bbox_height_px': 20, 'bbox_width_px': 300},
    {'text': 'Trans Fat (g) 0.0 0.0', 'confidence': 0.96, 'bbox': [0,140,250,160], 'bbox_height_px': 20, 'bbox_width_px': 250},
    {'text': 'Carbohydrate (g) 61.2 15.3', 'confidence': 0.95, 'bbox': [0,160,310,180], 'bbox_height_px': 20, 'bbox_width_px': 310},
    {'text': 'Total Sugar (g) 2.4 0.6', 'confidence': 0.94, 'bbox': [0,180,270,200], 'bbox_height_px': 20, 'bbox_width_px': 270},
    {'text': 'Protein (g) 8.4 2.1', 'confidence': 0.96, 'bbox': [0,200,220,220], 'bbox_height_px': 20, 'bbox_width_px': 220},
    {'text': 'Sodium (mg) 1670 418', 'confidence': 0.95, 'bbox': [0,220,230,240], 'bbox_height_px': 20, 'bbox_width_px': 230},
    {'text': '% RDA per serving', 'confidence': 0.93, 'bbox': [0,240,230,260], 'bbox_height_px': 20, 'bbox_width_px': 230},
]

result = LegalMetrologyRulesEngine.evaluate_package(maggi_tokens)
print("VERDICT:", result['overall_verdict'])
print("PANEL:", result['panel_type'])
print("Checks:", result['compliant_count'], "compliant |", result['severe_violations_count'], "severe |", result['minor_infractions_count'], "minor")
print()
STATUS_ICONS = {"COMPLIANT": "OK ", "SEVERE_VIOLATION": "!!! ", "MINOR_INFRACTION": "!!  ", "INFORMATIONAL": "--- ", "NOT_APPLICABLE": "N/A "}
for r in result['rule_checks']:
    icon = STATUS_ICONS.get(r.status, "?   ")
    title_short = r.rule_title[:55]
    print(icon + title_short)
    if r.extracted_value:
        print("    Got:", r.extracted_value[:50], "| Need:", r.required_value[:50])
