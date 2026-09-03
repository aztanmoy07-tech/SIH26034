import re
import math
from typing import List, Tuple, Dict
from models import AiPayload, EntityTag, Violation, VerdictState, InspectionReport, ExtractedToken

class ComplianceService:
    
    def __init__(self):
        self.forbidden_words = ["when packed", "approximately", "approx", "minimum", "min.", "about"]

    def _get_tokens(self, payload: AiPayload, tag: EntityTag) -> List[ExtractedToken]:
        return [t for t in payload.tokens if t.entity_tag == tag]

    def verify_rule_6(self, payload: AiPayload) -> List[Violation]:
        """
        Rule Set 1 & 2: Mandatory Declarations & Forbidden Words
        """
        violations = []
        
        # 1.1 & 1.2 Manufacturer & Generic Name
        if not self._get_tokens(payload, EntityTag.B_MANUFACTURER):
            violations.append(Violation(rule_section="Rule 6(1)(a)", description="Manufacturer/Packer details missing.", severity=VerdictState.SEVERE_VIOLATION))
        else:
            # Check for 6-digit PIN code in manufacturer text
            mfg_text = " ".join([t.text for t in self._get_tokens(payload, EntityTag.B_MANUFACTURER)])
            if not re.search(r'\b\d{6}\b', mfg_text):
                violations.append(Violation(rule_section="Rule 6(1)(a)", description="Manufacturer address missing 6-digit PIN code.", severity=VerdictState.SEVERE_VIOLATION))

        if not self._get_tokens(payload, EntityTag.B_GENERIC_NAME):
            violations.append(Violation(rule_section="Rule 6(1)(b)", description="Generic/Common name missing. Brand name alone is insufficient.", severity=VerdictState.SEVERE_VIOLATION))

        # 1.3 & Rule Set 2: Net Quantity & Forbidden Words
        net_qty_tokens = self._get_tokens(payload, EntityTag.B_NET_QTY)
        if not net_qty_tokens:
            violations.append(Violation(rule_section="Rule 6(1)(c)", description="Net Quantity missing.", severity=VerdictState.SEVERE_VIOLATION))
        else:
            qty_text = " ".join([t.text.lower() for t in net_qty_tokens])
            for word in self.forbidden_words:
                if word in qty_text:
                    violations.append(Violation(rule_section="Rule 11 & 12", description=f"Forbidden qualifier '{word}' used with Net Quantity.", severity=VerdictState.SEVERE_VIOLATION))
            if not re.search(r'\d+\s*(g|kg|ml|l)\b', qty_text, re.IGNORECASE):
                violations.append(Violation(rule_section="Rule 6(1)(c)", description="Net Quantity not in standard SI units (g, kg, ml, L).", severity=VerdictState.SEVERE_VIOLATION))

        # 1.4 Date of Mfg
        if not self._get_tokens(payload, EntityTag.B_MFG_DATE):
            violations.append(Violation(rule_section="Rule 6(1)(d)", description="Month/Year of Manufacture missing.", severity=VerdictState.SEVERE_VIOLATION))

        # 1.5 Retail Sale Price (MRP)
        mrp_tokens = self._get_tokens(payload, EntityTag.B_MRP)
        if not mrp_tokens:
            violations.append(Violation(rule_section="Rule 6(1)(e)", description="MRP declaration missing.", severity=VerdictState.SEVERE_VIOLATION))
        else:
            mrp_text = " ".join([t.text.lower() for t in mrp_tokens])
            # Strict format check as requested
            valid_mrp = ("inclusive of all taxes" in mrp_text or "incl., of all taxes" in mrp_text) and ("mrp" in mrp_text or "maximum retail price" in mrp_text or "max. retail price" in mrp_text)
            if not valid_mrp:
                violations.append(Violation(rule_section="Rule 6(1)(e)", description="MRP formatting strictly invalid. Must include 'inclusive of all taxes'.", severity=VerdictState.SEVERE_VIOLATION))

        # 1.6 Consumer Care
        care_tokens = self._get_tokens(payload, EntityTag.B_CONSUMER_CARE)
        if not care_tokens:
            violations.append(Violation(rule_section="Rule 6(1)(g)", description="Consumer Care details missing.", severity=VerdictState.SEVERE_VIOLATION))
        else:
            care_text = " ".join([t.text for t in care_tokens])
            has_phone = bool(re.search(r'\d{10}', care_text))
            has_email = bool(re.search(r'\S+@\S+\.\S+', care_text))
            if not (has_phone and has_email):
                violations.append(Violation(rule_section="Rule 6(1)(g)", description="Consumer Care must include both a 10-digit phone and a valid email.", severity=VerdictState.MINOR_VIOLATION))

        return violations

    def verify_schedule_2_dimensions(self, payload: AiPayload) -> Tuple[List[Violation], float]:
        """
        Rule Set 3: Spatial Mathematics (Rules 7 & 9 / Schedule II)
        """
        violations = []
        d = payload.dimensions
        pdp_area = 0.0

        # 3.1 Calculate PDP Area
        if d.shape == "rectangular":
            pdp_area = d.height_cm * d.width_cm
        elif d.shape == "cylindrical":
            pdp_area = 0.40 * d.height_cm * d.circumference_cm
        elif d.shape == "irregular":
            pdp_area = 0.40 * d.surface_area_cm2

        # 3.2 Determine Required Min Height
        req_height_mm = 1.0
        if pdp_area > 2500:
            req_height_mm = 6.0
        elif pdp_area > 500:
            req_height_mm = 4.0
        elif pdp_area > 100:
            req_height_mm = 2.0

        # Evaluate against tokens
        for token in payload.tokens:
            if token.entity_tag in [EntityTag.B_NET_QTY, EntityTag.B_MRP, EntityTag.B_MFG_DATE]:
                actual_height_mm = token.bbox_height_px * payload.px_to_mm_ratio
                if actual_height_mm < req_height_mm:
                    violations.append(Violation(
                        rule_section="Schedule II",
                        description=f"{token.entity_tag.value} height {actual_height_mm:.1f}mm is less than required {req_height_mm}mm for PDP {pdp_area:.1f}cm2.",
                        severity=VerdictState.MINOR_VIOLATION
                    ))
                
                # 3.3 Spatial Clearances on Net Qty
                if token.entity_tag == EntityTag.B_NET_QTY:
                    req_top_bottom = token.bbox_height_px
                    req_left_right = 2 * token.bbox_height_px
                    if (token.clearance_top_px < req_top_bottom or token.clearance_bottom_px < req_top_bottom or
                        token.clearance_left_px < req_left_right or token.clearance_right_px < req_left_right):
                        violations.append(Violation(
                            rule_section="Rule 9(3)",
                            description="Net Quantity spatial clearance insufficient (requires 1x height above/below, 2x left/right).",
                            severity=VerdictState.MINOR_VIOLATION
                        ))

        return violations, pdp_area

    def verify_fssai_overlap(self, payload: AiPayload) -> List[Violation]:
        """
        Rule Set 4: FSSAI Interfacing Regulations
        """
        violations = []
        
        # 4.1 Veg/Non-Veg
        has_veg = bool(self._get_tokens(payload, EntityTag.B_VEG_MARK))
        has_nonveg = bool(self._get_tokens(payload, EntityTag.B_NON_VEG_MARK))
        if not (has_veg or has_nonveg):
            violations.append(Violation(rule_section="FSSAI Rules", description="Vegetarian/Non-Vegetarian geometric symbol missing.", severity=VerdictState.SEVERE_VIOLATION))

        # 4.2 FSSAI License Format
        lic_tokens = self._get_tokens(payload, EntityTag.B_FSSAI_LIC)
        if not lic_tokens:
            violations.append(Violation(rule_section="FSSAI Rules", description="FSSAI License Number missing.", severity=VerdictState.SEVERE_VIOLATION))
        else:
            lic_text = " ".join([t.text for t in lic_tokens])
            if not re.search(r'\b[12]\d{13}\b', lic_text):
                violations.append(Violation(rule_section="FSSAI Rules", description="Invalid FSSAI License format. Must be 14 digits starting with 1 or 2.", severity=VerdictState.SEVERE_VIOLATION))
                
        return violations

    def verify_ecommerce_parity(self, payload: AiPayload) -> List[Violation]:
        """
        Rule Set 5: E-Commerce Digital Parity
        """
        violations = []
        if payload.digital_net_weight:
            physical_qty_tokens = self._get_tokens(payload, EntityTag.B_NET_QTY)
            if physical_qty_tokens:
                phys_text = " ".join([t.text for t in physical_qty_tokens]).lower()
                if payload.digital_net_weight.lower() not in phys_text:
                    violations.append(Violation(rule_section="Rule 18", description=f"E-commerce parity failure. Digital: {payload.digital_net_weight}, Physical does not match.", severity=VerdictState.SEVERE_VIOLATION))
        
        if payload.digital_country_of_origin == "MISSING":
             violations.append(Violation(rule_section="Rule 6(10A) 2026", description="E-commerce listing missing Country of Origin filter.", severity=VerdictState.SEVERE_VIOLATION))

        return violations

    def generate_enforcement_action(self, violations: List[Violation]) -> Tuple[VerdictState, str]:
        """
        Rule Set 6: Enforcement Logic (Jan Vishwas Act Routing)
        """
        if not violations:
            return VerdictState.COMPLIANT, "Packaging is fully compliant. No action required."
        
        is_severe = any(v.severity == VerdictState.SEVERE_VIOLATION for v in violations)
        
        if is_severe:
            return VerdictState.SEVERE_VIOLATION, "Prosecution Report / Violation Summary generated under Section 36 (Fines up to Rs. 1,00,000)."
        else:
            return VerdictState.MINOR_VIOLATION, "Digital Improvement Notice generated (Jan Vishwas Act). 15-30 days granted to rectify."

    def process_inspection(self, payload: AiPayload) -> InspectionReport:
        """
        Orchestrates all rule sets.
        """
        all_violations = []
        all_violations.extend(self.verify_rule_6(payload))
        
        spatial_violations, pdp_area = self.verify_schedule_2_dimensions(payload)
        all_violations.extend(spatial_violations)
        
        all_violations.extend(self.verify_fssai_overlap(payload))
        all_violations.extend(self.verify_ecommerce_parity(payload))

        verdict, action = self.generate_enforcement_action(all_violations)

        return InspectionReport(
            scan_id=payload.scan_id,
            verdict=verdict,
            violations=all_violations,
            action_required=action,
            pdp_area_cm2=pdp_area
        )
