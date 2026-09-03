import os
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
from cv_pipeline import ImagePreprocessor, PackageExtractor
from legal_rules import LegalMetrologyRulesEngine

def run_bulk_scan(input_dir="test_samples", output_csv="bulk_compliance_report.csv"):
    print(f"[*] Starting Winter Arc Bulk Compliance Scan on {input_dir}...")
    
    rules_engine = LegalMetrologyRulesEngine()
    
    image_paths = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print(f"[!] No images found in {input_dir}")
        return

    report_data = []

    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Scanning: {os.path.basename(img_path)}...")
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # 1. OCR Extraction
            tokens = PackageExtractor.run_ocr(img)
            # Combine tokens for the legacy rules engine API which expects a single string or uses it directly
            # Wait, legal_rules.py evaluate_all expects full text or tokens? 
            # In web_dashboard.py it does: ocr_tokens = PackageExtractor.run_ocr(img), combined_text = " ".join([t['text'] for t in ocr_tokens])
            combined_text = " ".join([t['text'] for t in tokens])
            
            # 2. Evaluate Rules
            results = rules_engine.evaluate_package(
                ocr_tokens=tokens,
                pdp_shape="rectangular",
                pdp_height_cm=15.0,
                pdp_width_cm=10.0,
                px_to_mm_ratio=0.1,
                is_food_commodity=True
            )
            
            # 3. Aggregate findings
            rule_checks = results["rule_checks"]
            passed = sum(1 for r in rule_checks if r.status == "COMPLIANT")
            failed = sum(1 for r in rule_checks if r.status in ["SEVERE_VIOLATION", "MINOR_INFRACTION"])
            total_rules = len(rule_checks)
            
            overall_status = "PASS" if failed == 0 else "FAIL"
            
            # Build string of failed rule IDs
            failed_rules = ", ".join([r.rule_id for r in rule_checks if r.status != "COMPLIANT"])
            
            # Sum up potential fines (rough heuristic for demonstration)
            total_fines = 0
            for r in rule_checks:
                if r.status != "COMPLIANT" and "Rs." in r.penalty_ref:
                    import re
                    fines = re.findall(r'Rs\.?\s*([\d,]+)', r.penalty_ref)
                    if fines:
                        total_fines += int(fines[0].replace(',', ''))

            report_data.append({
                "Filename": os.path.basename(img_path),
                "Status": overall_status,
                "Rules Checked": total_rules,
                "Rules Passed": passed,
                "Rules Failed": failed,
                "Failed Rules List": failed_rules,
                "Est. Max Penalty (INR)": total_fines
            })
            
        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")
            report_data.append({
                "Filename": os.path.basename(img_path),
                "Status": "ERROR",
                "Rules Checked": 0,
                "Rules Passed": 0,
                "Rules Failed": 0,
                "Failed Rules List": str(e),
                "Est. Max Penalty (INR)": 0
            })

    # Save to CSV
    df = pd.DataFrame(report_data)
    df.to_csv(output_csv, index=False)
    print(f"\n[OK] Bulk scan complete! Evaluated {len(image_paths)} products.")
    print(f"[*] Comprehensive report saved to: {output_csv}")
    
    # Print summary
    fails = len(df[df['Status'] == 'FAIL'])
    passes = len(df[df['Status'] == 'PASS'])
    total_penalty = df['Est. Max Penalty (INR)'].sum()
    print("="*50)
    print("               WINTER ARC SCAN SUMMARY               ")
    print("="*50)
    print(f"Total Products : {len(image_paths)}")
    print(f"Compliant      : {passes}")
    print(f"Non-Compliant  : {fails}")
    print(f"Total Exposure : Rs. {total_penalty:,}")
    print("="*50)

if __name__ == "__main__":
    run_bulk_scan()
