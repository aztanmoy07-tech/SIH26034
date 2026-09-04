"""
=============================================================================
MetriGuard — Ultra-Detailed Smart Rules Engine v3.0
Covers Legal Metrology (PC) Rules 2011 + FSSAI Labelling Regs 2020 fully.
Panel-aware: runs correct checks based on which face of label is in view.
=============================================================================
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RuleCheckResult:
    rule_id: str
    rule_title: str
    statutory_ref: str
    status: str  # COMPLIANT | MINOR_INFRACTION | SEVERE_VIOLATION | NOT_APPLICABLE | INFORMATIONAL
    extracted_text: str
    explanation: str
    what_was_checked: str = ""          # Describes the specific criteria
    extracted_value: str = ""           # The actual value found
    required_value: str = ""            # What the rule requires
    remedy_notice: str = ""
    penalty_ref: str = ""
    confidence: float = 1.0


# ============================================================
# PANEL CLASSIFIER
# ============================================================

NUTRITIONAL_KEYWORDS = [
    "calories", "calorie", "protein", "carbohydrate", "fat", "sodium", "cholesterol",
    "nutritional", "nutrition facts", "nutritive value", "per 100g", "per 100 g",
    "per serving", "daily value", "% dv", "minerals", "vitamins", "dietary fibre",
    "dietary fiber", "sugar", "trans fat", "saturated fat", "energy kcal", "energy kj",
    "calcium", "iron", "vitamin a", "vitamin c", "vitamin d", "riboflavin", "niacin",
    "thiamine", "folate", "total fat", "total carb", "added sugars", "monounsaturated",
    "polyunsaturated", "omega", "serving size", "servings per", "% rda"
]

INGREDIENT_KEYWORDS = [
    "ingredients:", "ingredients :", "contains:", "contains :", "derived from", "wheat flour",
    "refined flour", "maida", "edible oil", "edible vegetable oil", "permitted",
    "antioxidant", "preservative", "acidity regulator", "flavouring agent", "flavoring agent",
    "emulsifier", "stabiliser", "stabilizer", "nature identical", "added flavour",
    "raising agent", "leavening", "invert syrup", "dextrose", "maltodextrin",
    "high fructose", "modified starch", "liquid glucose", "yeast extract", "msg",
    "monosodium glutamate"
]

FRONT_PANEL_KEYWORDS = [
    "mrp", "m.r.p", "max retail", "rs.", "₹", "mfg", "manufactured", "packed",
    "marketed", "imported", "net", "net wt", "net qty", "net quantity", "fssai",
    "consumer", "helpline", "best before", "use by", "expiry", "date of mfg",
    "date of manufacture", "unit sale price", "usp", "country of origin"
]

BARCODE_KEYWORDS = ["barcode", "ean", "upc", "gtin", "scan here", "qr code"]

ALLERGEN_LIST = [
    "wheat", "gluten", "milk", "dairy", "egg", "peanut", "groundnut", "tree nuts",
    "soy", "soya", "fish", "shellfish", "crustacean", "sesame", "mustard", "celery",
    "sulphite", "sulfite", "lupin", "mollusc"
]

# FSSAI Schedule I — mandatory nutrient declaration (per 100g)
FSSAI_SCHEDULE_I_NUTRIENTS = {
    "energy": {"unit": "kcal or kJ", "pattern": r"energy\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*(kcal|kj|cal)"},
    "protein": {"unit": "g", "pattern": r"protein\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "carbohydrate": {"unit": "g", "pattern": r"(carbohydrate|carbs|carbo)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "total_fat": {"unit": "g", "pattern": r"(total fat|fat total|fat)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "saturated_fat": {"unit": "g", "pattern": r"(saturated fat|sat fat|saturated)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "trans_fat": {"unit": "g", "pattern": r"(trans fat|trans fatty)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "sugar": {"unit": "g", "pattern": r"(sugar|sugars|total sugar)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
    "sodium": {"unit": "mg", "pattern": r"sodium\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*mg"},
    "dietary_fibre": {"unit": "g", "pattern": r"(dietary fibre|dietary fiber|fibre)\s*[:\|]?\s*(\d+(?:\.\d+)?)\s*g"},
}

# ============================================================
# FSSAI 2020 Reference RDA Values (Regulation 5(3))
# ============================================================
FSSAI_RDA = {
    "energy_kcal": 2000,
    "total_fat_g": 67,
    "saturated_fat_g": 22,
    "trans_fat_g": 2,
    "added_sugar_g": 50,
    "sodium_mg": 2000,  # = 5g salt
}

# Legal Metrology Rule 7, Table I — Official 5-tier font height schedule
# Source: LM(PC) Rules 2011, Rule 7, Table I
SCHEDULE_II = [
    (50,   1.0),    # PDP ≤ 50 cm²    → min 1.0 mm
    (100,  1.5),    # PDP ≤ 100 cm²   → min 1.5 mm
    (200,  2.0),    # PDP ≤ 200 cm²   → min 2.0 mm
    (500,  2.5),    # PDP ≤ 500 cm²   → min 2.5 mm
    (float('inf'), 4.0),  # PDP > 500 cm² → min 4.0 mm
]


def get_min_font_mm(area_cm2: float) -> float:
    for threshold, min_mm in SCHEDULE_II:
        if area_cm2 <= threshold:
            return min_mm
    return 6.0


def classify_panel(tokens: List[Dict], combined_text: str) -> Dict[str, Any]:
    lower = combined_text.lower()
    nutri_hits = sum(1 for kw in NUTRITIONAL_KEYWORDS if kw in lower)
    ingr_hits = sum(1 for kw in INGREDIENT_KEYWORDS if kw in lower)
    front_hits = sum(1 for kw in FRONT_PANEL_KEYWORDS if kw in lower)
    barcode_hits = sum(1 for kw in BARCODE_KEYWORDS if kw in lower)

    if nutri_hits >= 2 and nutri_hits >= front_hits:
        panel = "NUTRITIONAL_BACK_PANEL"
    elif ingr_hits >= 2 and ingr_hits >= front_hits:
        panel = "INGREDIENT_BACK_PANEL"
    elif barcode_hits >= 1 and front_hits <= 1:
        panel = "BARCODE_PANEL"
    else:
        panel = "FRONT_PDP"

    return {
        "panel_type": panel,
        "nutri_hits": nutri_hits,
        "ingr_hits": ingr_hits,
        "front_hits": front_hits,
        "panel_description": {
            "NUTRITIONAL_BACK_PANEL": "Nutritional Table / Back Panel (Not the Principal Display Panel)",
            "INGREDIENT_BACK_PANEL": "Ingredient List / Allergen Panel (Side or Back Panel)",
            "FRONT_PDP": "Principal Display Panel — Front Face of Package",
            "BARCODE_PANEL": "Barcode / QR Panel (Informational Only)"
        }.get(panel, "Unknown Panel"),
        "audit_scope": {
            "NUTRITIONAL_BACK_PANEL": "FSSAI Labelling Regulations 2020 — Nutritional Table Rules",
            "INGREDIENT_BACK_PANEL": "FSSAI Labelling Regs 2020 — Ingredient & Allergen Declaration Rules",
            "FRONT_PDP": "Legal Metrology (PC) Rules 2011 — Full Rule 6 Compliance",
            "BARCODE_PANEL": "Informational Only — No compliance rules apply to barcode panels"
        }.get(panel, "")
    }


class LegalMetrologyRulesEngine:
    FORBIDDEN_QUALIFIERS = ["when packed", "approximately", "approx", "minimum", "min.", "about", "upto", "up to"]

    @classmethod
    def evaluate_package(cls,
        ocr_tokens: List[Dict[str, Any]],
        pdp_shape: str = "rectangular",
        pdp_height_cm: float = 15.0,
        pdp_width_cm: float = 10.0,
        pdp_circumference_cm: float = 0.0,
        pdp_surface_area_cm2: float = 0.0,
        px_to_mm_ratio: float = 0.1,
        is_food_commodity: bool = True
    ) -> Dict[str, Any]:

        results: List[RuleCheckResult] = []
        combined = " ".join(t["text"] for t in ocr_tokens)
        lower = combined.lower()

        # ─── STEP 1: Classify panel ───────────────────────────────────────────
        panel_info = classify_panel(ocr_tokens, combined)
        panel_type = panel_info["panel_type"]
        is_front = (panel_type == "FRONT_PDP")
        is_nutri = (panel_type == "NUTRITIONAL_BACK_PANEL")
        is_ingr = (panel_type == "INGREDIENT_BACK_PANEL")

        # Add panel detection info card
        results.append(RuleCheckResult(
            rule_id="PANEL_DETECTION",
            rule_title="Label Section / Panel Identified by AI",
            statutory_ref="Legal Metrology Act, 2009 — Multi-Panel Label Architecture",
            status="INFORMATIONAL",
            extracted_text=panel_info["panel_description"],
            what_was_checked="OCR text analysed for panel-type keywords. Front PDP keywords: MRP, Mfg, FSSAI, Net Qty. Nutritional keywords: Calories, Fat, Protein, Carbs etc.",
            explanation=(
                f"MetriGuard detected this image shows the '{panel_info['panel_description']}'. "
                f"Compliance scope applied: {panel_info['audit_scope']}."
            )
        ))

        # ─── NUTRITIONAL BACK PANEL — Full FSSAI Schedule I checks ───────────
        if is_nutri:
            cls._check_nutritional_table(results, combined, lower, ocr_tokens)

        # ─── INGREDIENT PANEL ─────────────────────────────────────────────────
        elif is_ingr:
            cls._check_ingredient_panel(results, combined, lower)

        # ─── BARCODE PANEL ────────────────────────────────────────────────────
        elif panel_type == "BARCODE_PANEL":
            results.append(RuleCheckResult(
                rule_id="BARCODE_INFO",
                rule_title="Barcode / QR Panel — No Compliance Rules Apply",
                statutory_ref="Legal Metrology (PC) Rules 2011",
                status="INFORMATIONAL",
                extracted_text="Barcode / QR Code region detected",
                explanation="Upload the Principal Display Panel (front face) for compliance audit."
            ))

        # ─── FRONT PDP — Full Rule 6 audit ────────────────────────────────────
        elif is_front:
            pdp_area = cls._calc_pdp_area(pdp_shape, pdp_height_cm, pdp_width_cm, pdp_circumference_cm, pdp_surface_area_cm2)
            min_font_mm = get_min_font_mm(pdp_area)
            cls._check_front_pdp(results, combined, lower, ocr_tokens, is_food_commodity, pdp_area, min_font_mm, px_to_mm_ratio)

        # ─── FINAL VERDICT ─────────────────────────────────────────────────────
        severe = sum(1 for r in results if r.status == "SEVERE_VIOLATION")
        minor = sum(1 for r in results if r.status == "MINOR_INFRACTION")
        compliant = sum(1 for r in results if r.status == "COMPLIANT")
        needs_review = sum(1 for r in results if r.status == "REQUIRES_MANUAL_REVIEW")
        pdp_area = pdp_height_cm * pdp_width_cm
        min_font_mm = get_min_font_mm(pdp_area)

        if not is_front:
            # Still show real violation count for nutritional/ingredient panels
            if severe > 0:
                verdict = "NUTRITIONAL_VIOLATION"
                headline = f"Nutritional Table Detected — {severe} Severe Violation(s) Found"
                description = (
                    f"This image shows the {panel_info['panel_description']}. "
                    f"FSSAI Schedule I audit found {severe} SEVERE VIOLATION(S) and {minor} minor infraction(s). "
                    f"Missing mandatory nutrients are liable under the FSS Act, 2006 — penalty up to Rs. 3,00,000 per missing declaration. "
                    f"Upload the Principal Display Panel (front face) for complete Legal Metrology (PC) Rules audit."
                )
            elif minor > 0:
                verdict = "NUTRITIONAL_MINOR"
                headline = f"Nutritional Table Detected — {minor} Minor Infraction(s) Found"
                description = (
                    f"This image shows the {panel_info['panel_description']}. "
                    f"FSSAI Schedule I audit found {minor} minor infraction(s) requiring correction. "
                    f"Upload the Principal Display Panel (front face) for full Legal Metrology compliance audit."
                )
            else:
                verdict = "PANEL_SCAN_ONLY"
                headline = f"{'Nutritional Table' if is_nutri else 'Ingredient Panel'} Detected — All Checked Items Compliant"
                description = (
                    f"This image shows the {panel_info['panel_description']}. "
                    f"FSSAI labelling checks were applied. For complete Legal Metrology (PC) Rules compliance, "
                    f"upload the Principal Display Panel (front face) of this package."
                )
        elif severe == 0 and minor == 0:
            verdict = "COMPLIANT"
            headline = "✅ Fully Compliant — Legal Metrology (PC) Rules, 2011"
            description = f"All {compliant} mandatory declarations verified and compliant. No regulatory action required."
        elif severe > 0:
            verdict = "SEVERE_VIOLATION"
            headline = "🔴 Prosecution / Seizure Notice Recommended"
            description = (
                f"Found {severe} severe violation(s) and {minor} minor infraction(s). "
                f"Liable for compounding or prosecution under Section 36 of the Legal Metrology Act, 2009. "
                f"Fine: Rs. 2,000 to Rs. 1,00,000 per violation (Jan Vishwas 2023 Amendment)."
            )
        else:
            verdict = "IMPROVEMENT_NOTICE"
            headline = "🟡 Digital Improvement Notice — Jan Vishwas Act, 2023"
            description = (
                f"Found {minor} minor/procedural infraction(s). Under the Jan Vishwas Act 2023, "
                f"a digital improvement notice is issued. Firm has 15–30 days to rectify before compounding."
            )

        return {
            "overall_verdict": verdict,
            "action_headline": headline,
            "action_description": description,
            "panel_type": panel_type,
            "panel_description": panel_info["panel_description"],
            "audit_scope": panel_info["audit_scope"],
            "pdp_area_cm2": round(pdp_area, 2),
            "min_font_size_mm": min_font_mm,
            "severe_violations_count": severe,
            "minor_infractions_count": minor,
            "needs_review_count": needs_review,
            "compliant_count": compliant,
            "rule_checks": results
        }

    # =========================================================================
    # NUTRITIONAL TABLE CHECKS (FSSAI Labelling Regulations 2020)
    # TOKEN-BASED SMART DETECTION — reads actual OCR tokens in order,
    # uses spatial proximity and nutrient-table context to avoid false positives.
    # =========================================================================
    @classmethod
    def _check_nutritional_table(cls, results, combined, lower, tokens):

        # ── Step 1: Build a clean token list sorted top-to-bottom ──────────────
        # Tokens are sorted vertically. We look for nutrient names and then check
        # that a numeric value appears NEARBY (within ~3 vertical lines = same row).
        sorted_tokens = sorted(tokens, key=lambda t: (t.get("bbox", [0,0,0,0])[1], t.get("bbox", [0,0,0,0])[0]))
        token_texts  = [t["text"].strip()  for t in sorted_tokens]
        token_bboxes = [t.get("bbox", [0,0,0,0]) for t in sorted_tokens]

        # ── Step 2: Helper — find a nutrient by name in token list ────────────
        def find_nutrient_in_tokens(name_variants, require_number=True):
            """
            Looks for any of the name_variants in token texts.
            Returns (matched_token_text, extracted_value) or (None, None).

            Strategy:
            1. Find token(s) whose text contains a name variant (case-insensitive word boundary).
            2. Look in the SAME ROW (±30px vertical) for a number token.
            3. If the number is there — COMPLIANT. If only the label but no value — still COMPLIANT
               (table header row; assume value is in the same row but OCR split it).
            4. If no matching token found at all — NOT DETECTED.
            """
            # Allow digits, O/o misreads, dashes, nil, zero
            NUMBER_RE = re.compile(r'(?:^|\s|:)([\dOo]+(?:[.,][\dOo]+)?|-|nil|zero)\s*(g|mg|kcal|kj|cal|%|ml)?\b', re.IGNORECASE)

            for i, text in enumerate(token_texts):
                tl = text.lower()
                import difflib
                
                # Word-boundary check: nutrient name must be a whole word/phrase
                for variant in name_variants:
                    # Allow partial suffix match but prevent substring matches in other words
                    # e.g. allow "protein3.4g" but prevent "fat" matching "saturated fat"
                    # We use [^a-z] instead of \b because \b treats numbers as word chars, failing on "Protein3.4g"
                    pat = re.compile(r'(?:^|[^a-z])' + re.escape(variant) + r'(?:[^a-z]|$)', re.IGNORECASE)
                    
                    # Fuzzy match logic
                    clean_tl = re.sub(r'[^a-z\s]', '', tl).strip()
                    is_match = False
                    
                    if pat.search(tl):
                        is_match = True
                    else:
                        # Fallback to fuzzy matching for OCR errors like "Prolein" or "prota"
                        words = clean_tl.split()
                        if len(variant.split()) == 1:
                            # 50% error margin allowed per user request
                            matches = difflib.get_close_matches(variant, words, n=1, cutoff=0.5)
                            if matches:
                                is_match = True
                        else:
                            # For multi-word variants (e.g. "trans fat")
                            # 50% error margin allowed
                            if difflib.SequenceMatcher(None, variant, clean_tl).ratio() > 0.5:
                                is_match = True

                    if is_match:
                        # Found the nutrient label token
                        label_y_center = (token_bboxes[i][1] + token_bboxes[i][3]) / 2
                        label_height = max(1, token_bboxes[i][3] - token_bboxes[i][1])
                        # Dynamic threshold: max of 35px or 1.2x the text height
                        threshold = max(35.0, label_height * 1.2)
                        
                        # Gather all tokens in the same horizontal band
                        row_texts = []
                        for j, bbox in enumerate(token_bboxes):
                            jy = (bbox[1] + bbox[3]) / 2
                            if abs(jy - label_y_center) <= threshold:
                                row_texts.append(token_texts[j])

                        row_combined = " ".join(row_texts)
                        num_match = NUMBER_RE.search(row_combined)

                        if num_match:
                            val = num_match.group(1).strip()
                            unit = num_match.group(2) if num_match.group(2) else ""
                            return text, f"{val} {unit}".strip()
                        elif not require_number:
                            return text, "Detected (value in adjacent column)"
                        else:
                            # Label exists but no number in same row —
                            # could be OCR split across lines; check next 2 tokens
                            next_tokens_text = " ".join(token_texts[i+1:i+4])
                            nm2 = NUMBER_RE.search(next_tokens_text)
                            if nm2:
                                val2 = nm2.group(1).strip()
                                unit2 = nm2.group(2) if nm2.group(2) else ""
                                return text, f"{val2} {unit2}".strip()
                            # Label found but completely no number anywhere close
                            # → treat as UNCERTAIN (not a hard violation)
                            return text, "—"
            return None, None

        def make_result(rule_id, title, ref, names, what_checked, req_value, severity_if_missing, require_number=True):
            label, value = find_nutrient_in_tokens(names, require_number=require_number)
            found = label is not None
            value_found = found and value and value != "—"

            if found and value_found:
                status = "COMPLIANT"
                expl = f"'{label}' detected with value '{value}'."
            elif found and not value_found:
                # Label row found but no numeric value — treat as UNCERTAIN not violation
                status = "REQUIRES_MANUAL_REVIEW"
                expl = f"'{label}' label found but numeric value could not be read clearly. Manual review needed."
            else:
                status = severity_if_missing
                expl = f"'{names[0].title()}' not found in the nutritional table. This is mandatory under FSSAI FSS Labelling Regulations 2020."

            return RuleCheckResult(
                rule_id=rule_id,
                rule_title=title,
                statutory_ref=ref,
                status=status,
                extracted_text=label or "Not Detected",
                what_was_checked=what_checked,
                extracted_value=value or "—",
                required_value=req_value,
                explanation=expl,
                penalty_ref="" if status == "COMPLIANT" else "FSS Act 2006 Section 52 — penalty up to Rs. 3,00,000 per missing declaration."
            )

        # ── Step 3: Check all mandatory FSSAI Schedule I nutrients ────────────

        results.append(make_result(
            "FSSAI_NI_ENERGY", "Energy Declaration (per 100g + per serving)",
            "FSSAI FSS (Labelling & Display) Regulations 2020 — Schedule I, Regulation 4",
            ["energy", "calorie", "calories", "energy (kcal)", "energy(kcal)", "cal", "energi", "enrgy", "enegy"],
            "Energy in kcal must be declared per 100g AND per serving/portion.",
            "Energy in kcal (or kJ) per 100g — mandatory in both columns",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_PROTEIN", "Protein Declaration (per 100g)",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I, Reg 5(3)",
            ["protein", "proteins", "protien", "protiens", "proein", "prtein", "protei", "protin"],
            "Protein content in grams per 100g must be declared in the nutrition table.",
            "Protein in grams per 100g (mandatory)",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_CARBS", "Carbohydrate (Total) Declaration",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I, Reg 5(3)",
            ["carbohydrate", "carbohydrates", "total carbohydrate", "carbs", "carbohydrat", "carbo hydrate", "carbo", "carbohydrt", "carb"],
            "Total carbohydrate in grams per 100g must be present.",
            "Total Carbohydrate in grams per 100g",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_SUGAR", "Total Sugars Sub-Declaration",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I (sub-item under Carbohydrate)",
            ["sugar", "sugars", "total sugar", "total sugars", "of which sugar", "sugr", "sugrars", "sugrs"],
            "Total sugars must be sub-declared under Total Carbohydrate in grams per 100g.",
            "Total Sugars in grams per 100g (sub-item of Carbohydrate)",
            "MINOR_INFRACTION"
        ))

        results.append(make_result(
            "FSSAI_NI_FAT", "Total Fat Declaration",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I, Reg 5(3)",
            ["total fat", "fat total", "fat (g)", "total fat (g)", "fat", "fats"],
            "Total fat in grams per 100g must be present in nutrition table.",
            "Total Fat in grams per 100g",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_SAT_FAT", "Saturated Fat Sub-Declaration",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I, Reg 5(3)",
            ["saturated fat", "sat fat", "saturated fatty", "of which saturated", "saturatd fat", "saturate fat", "saturatd"],
            "Saturated Fat must be sub-declared under Total Fat per 100g.",
            "Saturated Fat in grams per 100g (sub-item of Total Fat)",
            "MINOR_INFRACTION"
        ))

        results.append(make_result(
            "FSSAI_NI_TRANS_FAT", "Trans Fat Declaration (Mandatory Even if 0g)",
            "FSSAI FSS (Labelling) Regs 2020 — Regulation 4(3) — Trans Fat mandatory",
            ["trans fat", "trans fatty acid", "trans fatty", "of which trans", "trns fat", "tran fat", "trns"],
            "Trans fatty acids in grams per 100g — mandatory since FSSAI 2020. Even if 0g, must be declared.",
            "Trans Fat in grams per 100g (0.0g must still be declared)",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_SODIUM", "Sodium Declaration (mg per 100g)",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I",
            ["sodium", "sodium (mg)", "salt", "sodum", "sodiun"],
            "Sodium must be declared in milligrams (mg) per 100g.",
            "Sodium in mg per 100g",
            "SEVERE_VIOLATION"
        ))

        results.append(make_result(
            "FSSAI_NI_FIBRE", "Dietary Fibre Declaration",
            "FSSAI FSS (Labelling) Regs 2020 — Schedule I (conditionally mandatory)",
            ["dietary fibre", "dietary fiber", "fibre", "fiber", "fibe", "fibr", "fibra"],
            "Dietary fibre is conditionally mandatory. Mandatory if product makes any fibre claim.",
            "Dietary Fibre in grams per 100g (mandatory if fibre claim made)",
            "MINOR_INFRACTION"
        ))

        # ── Step 4: Structural checks (column layout, serving size, %RDA) ──────

        per_100g_found = any(
            re.search(r'\bper\s*100\b', t.lower()) for t in token_texts
        )
        per_serving_found = any(
            re.search(r'\bper\s*serv|\bserving\b|\b%\s*rda\b|\brda\b', t.lower()) for t in token_texts
        )

        results.append(RuleCheckResult(
            rule_id="FSSAI_NI_COLUMNS",
            rule_title="Dual-Column Declaration (Per 100g + Per Serving)",
            statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 4(1)(b) — Two-column format",
            status="COMPLIANT" if (per_100g_found and per_serving_found)
                   else ("MINOR_INFRACTION" if (per_100g_found or per_serving_found)
                   else "SEVERE_VIOLATION"),
            extracted_text=f"per 100g: {'Found' if per_100g_found else 'MISSING'} | per serving/%RDA: {'Found' if per_serving_found else 'MISSING'}",
            what_was_checked="FSSAI 2020 mandates nutritional values in TWO columns: per 100g/ml AND per serving/portion.",
            extracted_value=("Both columns" if (per_100g_found and per_serving_found)
                            else ("Only per 100g column" if per_100g_found
                            else ("Only per serving column" if per_serving_found else "Neither column found"))),
            required_value="Both 'per 100g' AND 'per serving' columns (Regulation 4(1)(b))",
            explanation=("Both required nutritional columns detected." if (per_100g_found and per_serving_found)
                        else "Missing one or both required columns. FSSAI mandates both 'per 100g' and 'per serving' columns."),
            remedy_notice="" if (per_100g_found and per_serving_found)
                           else "Add missing column (per 100g or per serving) to the nutritional table."
        ))

        serving_label, serving_val = find_nutrient_in_tokens(
            ["serving size", "serving sizes", "per serving size"], require_number=True
        )
        results.append(RuleCheckResult(
            rule_id="FSSAI_NI_SERVING_SIZE",
            rule_title="Serving Size Declaration",
            statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 4(1)(a)",
            status="COMPLIANT" if (serving_label and serving_val and serving_val != "—") else "MINOR_INFRACTION",
            extracted_text=f"{serving_label} {serving_val}" if serving_label else "Not Detected",
            what_was_checked="Serving size in grams or ml must be stated above or within the nutritional table.",
            extracted_value=serving_val or "—",
            required_value="Serving size in g or ml (mandatory per Regulation 4(1)(a))",
            explanation="Serving size declared." if (serving_label and serving_val and serving_val != "—")
                        else "Serving size not found or value not readable. Must be declared before the nutritional table.",
            remedy_notice="" if (serving_label and serving_val and serving_val != "—")
                           else "Add 'Serving Size: X g' at the top of the nutritional table."
        ))

        rda_tokens = [t.lower() for t in token_texts]
        has_rda = any("rda" in t or "% daily" in t or "daily value" in t for t in rda_tokens)
        results.append(RuleCheckResult(
            rule_id="FSSAI_NI_RDA",
            rule_title="% RDA / % Daily Value Column",
            statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Schedule I, Footnote",
            status="COMPLIANT" if has_rda else "MINOR_INFRACTION",
            extracted_text="% RDA column detected" if has_rda else "% RDA column not found",
            what_was_checked="% Recommended Daily Allowance (%RDA) column must be shown. Footnote: '*% RDA for an average adult (2000 kcal/day)' mandatory.",
            extracted_value="Present" if has_rda else "Absent",
            required_value="% RDA column with footnote explaining basis (2000 kcal adult)",
            explanation="% RDA column found." if has_rda
                        else "% RDA column not detected. Must include footnote '*Percent RDA for an average adult (2000 kcal/day)'.",
            remedy_notice="" if has_rda else "Add %RDA column. Include footnote '*Percent RDA for an average adult (2000 kcal/day)'."
        ))

        # ── Step 5: Allergen advisory ──────────────────────────────────────────
        found_allergens = [a for a in ALLERGEN_LIST if a in lower]
        has_allergen_box = bool(re.search(r'\b(contains|allergen|allergy|may contain)\b', lower))
        if found_allergens:
            results.append(RuleCheckResult(
                rule_id="FSSAI_ALLERGEN",
                rule_title="Allergen Advisory Declaration",
                statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 5 (Allergen Declaration)",
                status="COMPLIANT" if has_allergen_box else "SEVERE_VIOLATION",
                extracted_text=f"Allergens in text: {', '.join(found_allergens[:5])}",
                what_was_checked="FSSAI 2020 mandates allergens be declared bold in ingredient list AND in a separate 'Contains:' advisory box.",
                extracted_value="Allergens present: " + ", ".join(found_allergens),
                required_value="Bold allergen text in ingredients + 'Contains: [allergen]' advisory",
                explanation="Allergen advisory box detected." if has_allergen_box
                            else f"Allergens ({', '.join(found_allergens[:3])}) detected but no 'Contains:' advisory box found.",
                penalty_ref="" if has_allergen_box else "FSS Act 2006 — penalty for non-declaration of allergens."
            ))



    # =========================================================================
    # INGREDIENT PANEL CHECKS
    # =========================================================================
    @classmethod
    def _check_ingredient_panel(cls, results, combined, lower):
        # Ingredient list completeness
        has_ingr_list = "ingredients" in lower or "ingredients:" in lower
        results.append(RuleCheckResult(
            rule_id="FSSAI_INGR_LIST",
            rule_title="Ingredient List Declaration",
            statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 3(1)",
            status="COMPLIANT" if has_ingr_list else "SEVERE_VIOLATION",
            extracted_text="Ingredient list header detected" if has_ingr_list else "Not found",
            what_was_checked="The word 'Ingredients:' must appear as a header before the ingredient list, in descending order of weight at the time of manufacture.",
            explanation="Ingredient list header found." if has_ingr_list else "Ingredient list header not found. 'Ingredients:' must appear in descending weight order.",
        ))

        # Allergens in bold
        found_allergens = [a for a in ALLERGEN_LIST if a in lower]
        has_allergen_advisory = "contains:" in lower or "allergen" in lower
        if found_allergens:
            results.append(RuleCheckResult(
                rule_id="FSSAI_INGR_ALLERGEN",
                rule_title="Allergen Declaration (Ingredient Panel)",
                statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 5",
                status="COMPLIANT" if has_allergen_advisory else "SEVERE_VIOLATION",
                extracted_text=f"Allergens in text: {', '.join(found_allergens[:6])}",
                what_was_checked="All 14 major allergens must be: (a) highlighted in bold/contrasting font in ingredient list, AND (b) declared in a separate 'Contains:' advisory box.",
                extracted_value=", ".join(found_allergens),
                required_value="Bold allergen text + 'Contains: X, Y' advisory",
                explanation="Allergen advisory found." if has_allergen_advisory else "Allergen advisory box ('Contains: ...') not found despite allergen-containing ingredients.",
                remedy_notice="" if has_allergen_advisory else "Add 'Contains: Wheat, Milk, [...]' advisory box per FSSAI 2020."
            ))

        # Additive declaration
        has_additive = any(w in lower for w in ["antioxidant", "preservative", "flavour", "emulsifier", "colour"])
        has_additive_ins = re.search(r'\(ins\s*\d+\)', lower)
        results.append(RuleCheckResult(
            rule_id="FSSAI_INGR_ADDITIVE",
            rule_title="Food Additive — INS Number Declaration",
            statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 3(3)",
            status="COMPLIANT" if (not has_additive or has_additive_ins) else "MINOR_INFRACTION",
            extracted_text=has_additive_ins.group(0) if has_additive_ins else "Additive detected without INS number",
            what_was_checked="Each food additive must be declared with its functional class name AND INS number, e.g. 'Antioxidant (INS 310)'.",
            explanation="INS number detected with additive declaration." if has_additive_ins else (
                "Food additive terms detected but INS numbers not confirmed. Each additive needs '(INS XXX)' format." if has_additive else "No food additives detected on this panel."),
            remedy_notice="" if (not has_additive or has_additive_ins) else "Format: 'Functional class (INS XXX)', e.g. 'Emulsifier (INS 322)'."
        ))

    # =========================================================================
    # FRONT PDP CHECKS (Legal Metrology PC Rules 2011 + FSSAI)
    # =========================================================================
    @classmethod
    def _check_front_pdp(cls, results, combined, lower, tokens, is_food, pdp_area, min_font_mm, px_per_mm):

        # ── Rule 6(1)(a): Manufacturer / Packer / Importer Details ────────────
        mfg_pattern = r'(mfg\.?|mfd\.?|manufactured|packed|marketed|imported)\s*(by|for)?[:\s]+([^,\.\n]{5,80})'
        mfg_match = re.search(mfg_pattern, combined, re.IGNORECASE)
        pin_match = re.search(r'\b([1-9][0-9]{5})\b', combined)
        phone_in_mfg = re.search(r'\b\d{10}\b', combined)

        if mfg_match and pin_match:
            status_mfg = "COMPLIANT"
            expl_mfg = f"Manufacturer/packer name and address with 6-digit PIN code ({pin_match.group(1)}) detected."
        elif mfg_match:
            status_mfg = "MINOR_INFRACTION"
            expl_mfg = "Manufacturer name found but complete address with 6-digit PIN code is missing."
        else:
            status_mfg = "SEVERE_VIOLATION"
            expl_mfg = "Manufacturer/Packer name and address NOT found. This is the most critical mandatory declaration."

        results.append(RuleCheckResult(
            rule_id="RULE_6_1_A",
            rule_title="Rule 6(1)(a) — Manufacturer / Packer / Importer Name & Address",
            statutory_ref="Rule 6(1)(a), Legal Metrology (PC) Rules, 2011 + Rule 6(1)(b) for importers",
            status=status_mfg,
            extracted_text=mfg_match.group(0)[:120].strip() if mfg_match else "Not Detected",
            what_was_checked="Full name + complete postal address with 6-digit PIN code of manufacturer, packer, or importer must be on PDP. If imported, importer's Indian address also required.",
            extracted_value=mfg_match.group(0)[:80].strip() if mfg_match else "—",
            required_value="Name + Full Address + 6-digit PIN (e.g. 'Mfg. by: ABC Foods, Okhla, New Delhi 110020')",
            explanation=expl_mfg,
            remedy_notice="" if status_mfg == "COMPLIANT" else "Add complete address with PIN code immediately below manufacturer name.",
            penalty_ref="" if status_mfg == "COMPLIANT" else "Section 36(1), Legal Metrology Act 2009 — Fine up to Rs. 25,000 / Imprisonment up to 1 year (repeat)."
        ))

        # ── Rule 6(1)(b): Generic / Common Name ────────────────────────────────
        generic_match = re.search(r'(generic name|commodity|common name|product)[:\s]+([A-Za-z\s]{3,50})', combined, re.IGNORECASE)
        results.append(RuleCheckResult(
            rule_id="RULE_6_1_B",
            rule_title="Rule 6(1)(b) — Generic / Common Name of Commodity",
            statutory_ref="Rule 6(1)(b), Legal Metrology (PC) Rules, 2011",
            status="COMPLIANT" if generic_match else "MINOR_INFRACTION",
            extracted_text=generic_match.group(0)[:80].strip() if generic_match else "Not Explicitly Found",
            what_was_checked="The generic/common name of the commodity must appear prominently below the brand name. Brand alone is insufficient (e.g., 'Maggi' without 'Noodles').",
            extracted_value=generic_match.group(2).strip() if generic_match else "—",
            required_value="Generic commodity name (e.g. 'Butter Biscuits', '2-Minute Noodles')",
            explanation="Generic name found." if generic_match else "Generic/common commodity name not explicitly found. Brand name alone does not satisfy this requirement.",
            remedy_notice="" if generic_match else "Add generic name prominently e.g. 'Instant Noodles' or 'Butter Biscuits' below brand name."
        ))

        # ── Rules 6(1)(c), 8, 11, 12: Net Quantity ─────────────────────────────
        net_match = re.search(
            r'(net\.?\s*(qty|quantity|wt|weight|vol|volume|content)?\.?)[:\s]*(\d+(?:\.\d+)?)\s*(g|kg|ml|l|litre|liter|gm|units?|n|nos?|tabs?)\b',
            combined, re.IGNORECASE)
        forbidden_found = [w for w in cls.FORBIDDEN_QUALIFIERS if w in lower]

        if forbidden_found:
            results.append(RuleCheckResult(
                rule_id="RULE_11_FORBID",
                rule_title="Rule 11 — Forbidden Qualifier on Net Quantity",
                statutory_ref="Rule 11(1), Legal Metrology (PC) Rules, 2011 — No deceptive qualifier",
                status="SEVERE_VIOLATION",
                extracted_text=f"Illegal qualifier found: '{forbidden_found[0]}'",
                what_was_checked=f"Forbidden qualifiers: {cls.FORBIDDEN_QUALIFIERS}. None may appear adjacent to net quantity declaration.",
                extracted_value=forbidden_found[0],
                required_value="No qualifier allowed — only bare numeric value + unit",
                explanation=f"Prohibited qualifier '{forbidden_found[0]}' found near net quantity. This misleads consumers about actual contents.",
                penalty_ref="Section 36(1) — Prosecution for misleading quantity declaration."
            ))
        else:
            results.append(RuleCheckResult(
                rule_id="RULE_6_1_C",
                rule_title="Rule 6(1)(c) + Rule 8 — Net Quantity in Standard Units",
                statutory_ref="Rules 6(1)(c), 8, 11, 12, Legal Metrology (PC) Rules, 2011",
                status="COMPLIANT" if net_match else "SEVERE_VIOLATION",
                extracted_text=net_match.group(0).strip() if net_match else "Not Detected",
                what_was_checked="Net quantity must: (a) use SI metric units (g/kg/ml/L), (b) appear near MRP, (c) no forbidden qualifier, (d) match Schedule III minimum pack size for category.",
                extracted_value=f"{net_match.group(3)} {net_match.group(4)}" if net_match else "—",
                required_value="Net Qty in g/kg/ml/L without any qualifier",
                explanation="Net quantity in SI units found." if net_match else "Net quantity declaration not found. This is a mandatory Prosecution-level violation.",
                penalty_ref="" if net_match else "Section 36(1) — Fine up to Rs. 1,00,000 (Jan Vishwas 2023 Amendment)."
            ))

        # ── Rule 6(1)(d): Date of Manufacture / Expiry ──────────────────────────
        date_match = re.search(
            r'(mfg|mfd|packed|pkd|manufactured|date of mfg|dom|best before|use by|expiry|exp|bb)\s*[:\s.]+([A-Za-z]{3,}\s*\d{4}|\d{1,2}[/\-]\d{4}|\d{2}[/\-.]\d{2}[/\-.]\d{2,4}|\d{1,2}\s*[A-Za-z]{3}\s*\d{2,4}|\d{1,2}\s*months?)',
            combined, re.IGNORECASE)
        results.append(RuleCheckResult(
            rule_id="RULE_6_1_D",
            rule_title="Rule 6(1)(d) — Month & Year of Manufacture / Packing / Import",
            statutory_ref="Rule 6(1)(d), Legal Metrology (PC) Rules, 2011",
            status="COMPLIANT" if date_match else "SEVERE_VIOLATION",
            extracted_text=date_match.group(0).strip() if date_match else "Not Detected",
            what_was_checked="Month and year of manufacture/packing/import. Acceptable formats: MM/YYYY, MMM YYYY, DD/MM/YYYY. Must include 'Mfg:' or 'Mfd:' prefix.",
            extracted_value=date_match.group(2).strip() if date_match else "—",
            required_value="Month + Year (e.g., 'Mfg: 08/2026' or 'Mfd: Aug 2026')",
            explanation="Manufacturing/packing date in compliant format found." if date_match else "Date of manufacture/packing not found. Mandatory for all pre-packed commodities.",
            penalty_ref="" if date_match else "Section 36(1) — Severe violation."
        ))

        # ── Rules 6(1)(e), 18: MRP ────────────────────────────────────────────
        mrp_match = re.search(
            r'(m\.?r\.?p\.?|max\.?\s*retail\s*price?|maximum\s*retail\s*price?)\s*[:\s]*(rs\.?|rs|inr|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
            combined, re.IGNORECASE)
        # Loosen MRP "inclusive of all taxes" to handle OCR errors like "incl. of all taxes"
        has_incl_tax = bool(re.search(r'(incl[a-z.]*\s*of\s*all\s*tax|all\s*tax[a-z]*\s*incl|inclusive)', lower))
        has_rupee = bool(re.search(r'(rs\.?|₹|inr)', combined, re.IGNORECASE))

        if mrp_match and has_incl_tax:
            mrp_status = "COMPLIANT"
            mrp_expl = f"MRP found with mandatory 'inclusive of all taxes' statutory phrase. Value: Rs. {mrp_match.group(3)}"
        elif mrp_match and not has_incl_tax:
            mrp_status = "SEVERE_VIOLATION"
            mrp_expl = "MRP value found but the mandatory statutory phrase 'inclusive of all taxes' is MISSING. This is a Prosecution-level violation."
        else:
            mrp_status = "SEVERE_VIOLATION"
            mrp_expl = "MRP declaration not found on this panel. All pre-packed commodities must display MRP."

        results.append(RuleCheckResult(
            rule_id="RULE_6_1_E_MRP",
            rule_title="Rule 6(1)(e) + Rule 18 — Maximum Retail Price (MRP)",
            statutory_ref="Rule 6(1)(e) & Rule 18, Legal Metrology (PC) Rules, 2011; Jan Vishwas Act 2023",
            status=mrp_status,
            extracted_text=mrp_match.group(0).strip() if mrp_match else "Not Detected",
            what_was_checked="MRP must: (a) appear as 'MRP Rs. X.XX (inclusive of all taxes)' or 'MRP ₹ X.XX (Incl. of all taxes)', (b) include the exact statutory phrase 'inclusive of all taxes', (c) not be under-printed vs actual charged price.",
            extracted_value=f"Rs. {mrp_match.group(3)}" if mrp_match else "—",
            required_value="'MRP Rs. X.XX (inclusive of all taxes)' — statutory phrase mandatory",
            explanation=mrp_expl,
            penalty_ref="" if mrp_status == "COMPLIANT" else "Section 36(1) — Penalty up to Rs. 1,00,000 (Jan Vishwas Amendment) or Prosecution."
        ))

        # ── Rule 6(1)(g) 2022: Consumer Care ─────────────────────────────────
        phone_match = re.search(r'(\b\d{10}\b|1800[\s\-]?\d{2,6}[\s\-]?\d{3,5})', combined)
        email_match = re.search(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,7}\b', combined)
        has_consumer_care = bool(re.search(r'(consumer|customer)\s*(care|helpline|support|grievance)', lower))

        if phone_match and email_match:
            cc_status = "COMPLIANT"
            cc_expl = f"Consumer care phone ({phone_match.group(0)}) and email ({email_match.group(0)}) both found."
        elif phone_match or email_match:
            cc_status = "MINOR_INFRACTION"
            found = phone_match.group(0) if phone_match else email_match.group(0)
            missing = "Email ID" if phone_match else "Phone / Toll-free number"
            cc_expl = f"Only {found} found. {missing} is also required per 2022 amendment."
        else:
            cc_status = "SEVERE_VIOLATION"
            cc_expl = "No consumer care contact details found. Both phone/toll-free AND email are mandatory."

        results.append(RuleCheckResult(
            rule_id="RULE_6_1_G",
            rule_title="Rule 6(1)(g) — Consumer Care Contact Details (2022 Amendment)",
            statutory_ref="Rule 6(1)(g), Legal Metrology (PC) Rules, 2011 (2022 Amendment adds email mandate)",
            status=cc_status,
            extracted_text=f"Phone: {phone_match.group(0) if phone_match else '—'} | Email: {email_match.group(0) if email_match else '—'}",
            what_was_checked="2022 Amendment mandates BOTH: (a) phone number or toll-free number (1800-XXX-XXXX format), AND (b) email ID for consumer grievances. Label 'Consumer Care' or 'Customer Helpline' must accompany.",
            extracted_value=f"{'✅ Phone' if phone_match else '❌ Phone'} | {'✅ Email' if email_match else '❌ Email'}",
            required_value="Both phone number AND email ID — both mandatory per 2022 Amendment",
            explanation=cc_expl,
            remedy_notice="" if cc_status == "COMPLIANT" else "Add missing contact. Format: 'Consumer Care: 1800-XXX-XXXX | consumer@brand.com'",
            penalty_ref="" if cc_status == "COMPLIANT" else "Section 36 — Improvement notice; repeated failure → prosecution."
        ))

        # ── Rule 6(11) 2022: Unit Sale Price (USP) ────────────────────────────
        usp_match = re.search(r'(unit\s*sale\s*price|usp)[:\s]*(rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(per\s*\d+)?\s*(g|kg|ml|l|unit|pc)', combined, re.IGNORECASE)
        results.append(RuleCheckResult(
            rule_id="RULE_6_11_USP",
            rule_title="Rule 6(11) — Unit Sale Price (2022 Amendment)",
            statutory_ref="Rule 6(11), Legal Metrology (PC) Rules 2011 (Inserted by 2022 Amendment)",
            status="COMPLIANT" if usp_match else "MINOR_INFRACTION",
            extracted_text=usp_match.group(0).strip() if usp_match else "Not Found",
            what_was_checked="Unit sale price (price per standard unit like per g, per ml, per kg) must be declared for easy consumer comparison. Format: 'Rs. X per g' or 'Rs. X per 100ml'.",
            extracted_value=usp_match.group(0) if usp_match else "—",
            required_value="'Unit Sale Price: Rs. X per g/ml/kg' adjacent to MRP",
            explanation="Unit Sale Price found and declared." if usp_match else "Unit Sale Price not found. Required since 2022 amendment for all pre-packed commodities.",
            remedy_notice="" if usp_match else "Calculate and print: 'Unit Sale Price: Rs. X per g' = MRP ÷ Net Weight."
        ))

        # ── Rule 6(1)(f) Country of Origin (Imports Only) ────────────────────
        import_match = re.search(r'(import|imported)', lower)
        coo_match = re.search(r'(country of origin|made in|product of)\s*[:\s]*([A-Za-z\s]{3,30})', combined, re.IGNORECASE)
        if import_match:
            results.append(RuleCheckResult(
                rule_id="RULE_6_1_F_COO",
                rule_title="Rule 6(1)(f) — Country of Origin (Imported Goods)",
                statutory_ref="Rule 6(1)(f), Legal Metrology (PC) Rules, 2011",
                status="COMPLIANT" if coo_match else "SEVERE_VIOLATION",
                extracted_text=coo_match.group(0).strip() if coo_match else "Not Found (but 'imported' detected)",
                what_was_checked="For imported pre-packed commodities, country of origin MUST be declared. Additionally, importer's name and Indian address required.",
                extracted_value=coo_match.group(2).strip() if coo_match else "—",
                required_value="'Country of Origin: [Country Name]' for all imported goods",
                explanation="Country of origin declared for imported product." if coo_match else "Import keyword detected but Country of Origin not declared. Mandatory for all imported goods.",
                penalty_ref="" if coo_match else "Section 36(1) — Prosecution."
            ))

        # ── FSSAI Checks (Food Commodities) ──────────────────────────────────
        if is_food:
            # Tolerate typical OCR errors where 0 is read as O, 1 as l or I
            fssai_lic = re.search(r'\b([12lI][\dO]{13})\b', combined, re.IGNORECASE)
            extracted_lic = fssai_lic.group(1).replace('O', '0').replace('l', '1').replace('I', '1').replace('o', '0') if fssai_lic else "—"
            
            results.append(RuleCheckResult(
                rule_id="FSSAI_14DIGIT",
                rule_title="FSSAI 14-Digit License Number",
                statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 2(1)(h); FSS Act 2006",
                status="COMPLIANT" if fssai_lic else "SEVERE_VIOLATION",
                extracted_text=f"FSSAI No. {extracted_lic}" if fssai_lic else "Not Detected",
                what_was_checked="Must be a 14-digit number starting with 1 or 2. Must appear near the FSSAI logo. Cannot be 13 or 15 digits.",
                extracted_value=extracted_lic,
                required_value="14-digit license starting with 1 or 2 (e.g., 10018011000142)",
                explanation=f"Valid 14-digit FSSAI license ({extracted_lic}) found." if fssai_lic else "FSSAI 14-digit license number not found. Mandatory for all food businesses.",
                penalty_ref="" if fssai_lic else "FSS Act 2006 Section 63 — selling without valid FSSAI license, penalty up to Rs. 5,00,000."
            ))

            # Veg/Non-veg symbol
            veg_match = re.search(r'(veg|vegetarian|non.?veg|non.?vegetarian|\[veg|green\s*dot|brown\s*triangle)', lower)
            results.append(RuleCheckResult(
                rule_id="FSSAI_VEG_NONVEG",
                rule_title="FSSAI Veg / Non-Veg Geometric Symbol",
                statutory_ref="FSSAI FSS (Labelling) Regs 2020 — Regulation 6 (Veg/Non-Veg Symbol)",
                status="COMPLIANT" if veg_match else "MINOR_INFRACTION",
                extracted_text="Veg / Non-Veg symbol detected" if veg_match else "Symbol not confirmed",
                what_was_checked="Must display either: (a) Green circle inside green square (Vegetarian), OR (b) Brown/maroon triangle inside brown square (Non-vegetarian). Symbol must be minimum 3mm.",
                extracted_value="Present" if veg_match else "Uncertain",
                required_value="Green square with circle (Veg) OR Brown square with triangle (Non-Veg), ≥ 3mm",
                explanation="Veg/Non-Veg symbol detected." if veg_match else "Veg/Non-Veg mark not confirmed by OCR — may be a graphic symbol not read as text. Verify visually.",
                remedy_notice="" if veg_match else "Ensure green or brown geometric symbol visible with minimum 3mm outer square."
            ))

        # ── Schedule II: Minimum Font Size ──────────────────────────────────
        avg_h_px = np.mean([t.get("bbox_height_px", 25) for t in tokens]) if tokens else 25.0
        est_font_mm = round(avg_h_px * px_per_mm, 2)
        results.append(RuleCheckResult(
            rule_id="SCHEDULE_II_FONT",
            rule_title=f"Schedule II — Minimum Numeral Height (PDP Area: {pdp_area:.0f} cm²)",
            statutory_ref="Schedule II, Legal Metrology (PC) Rules, 2011 (as amended)",
            status="COMPLIANT" if est_font_mm >= min_font_mm else "MINOR_INFRACTION",
            extracted_text=f"Estimated average character height: ~{est_font_mm} mm",
            what_was_checked=f"Schedule II minimum numeral height for PDP area {pdp_area:.0f} cm² is {min_font_mm} mm. All mandatory declarations (MRP, Net Qty, Mfg date) must meet this minimum.",
            extracted_value=f"{est_font_mm} mm (estimated from OCR bounding boxes)",
            required_value=f"≥ {min_font_mm} mm for PDP area {pdp_area:.0f} cm²",
            explanation=f"Estimated font height ~{est_font_mm} mm {'meets' if est_font_mm >= min_font_mm else 'FAILS'} the Schedule II minimum of {min_font_mm} mm.",
            remedy_notice="" if est_font_mm >= min_font_mm else f"Increase numeral height to at least {min_font_mm} mm for mandatory declarations."
        ))

    @staticmethod
    def _calc_pdp_area(shape, h, w, circ, surf) -> float:
        if shape == "cylindrical":
            return 0.40 * h * (circ if circ else w * 3.1416)
        elif shape == "irregular":
            return 0.40 * (surf if surf else h * w * 2)
        return h * w  # rectangular default
