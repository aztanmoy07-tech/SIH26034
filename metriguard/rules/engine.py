from typing import List, Dict, Any
import re
from datetime import datetime

from ..domain.schemas import (
    InspectionCase, PanelType, Finding, FindingStatus,
    HumanReviewState, Confidence, EvidenceCrop
)

class RulesRegistry:
    """Canonical registry for statutory rules."""
    
    RULES = {
        "LM_PC_6_1_e": {
            "title": "Maximum Retail Price (MRP) Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(e)",
            "severity_if_failed": FindingStatus.SEVERE_VIOLATION,
            "remediation": "Declare MRP inclusive of all taxes clearly on the front/side panel.",
            "penalty_reference": "Section 36(1) LM Act 2009"
        },
        "LM_PC_6_1_a": {
            "title": "Manufacturer Details & PIN Code",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(a)",
            "severity_if_failed": FindingStatus.SEVERE_VIOLATION,
            "remediation": "Provide full address of the manufacturer/packer including a 6-digit PIN code.",
            "penalty_reference": "Section 36(1) LM Act 2009"
        },
        "LM_PC_6_1_c": {
            "title": "Net Quantity Declaration",
            "statutory_ref": "Legal Metrology (Packaged Commodities) Rules, 2011, Rule 6(1)(c)",
            "severity_if_failed": FindingStatus.SEVERE_VIOLATION,
            "remediation": "Declare net quantity using standard units of weight, measure, or number without qualifiers.",
            "penalty_reference": "Section 36(1) LM Act 2009"
        }
    }

class CanonicalRulesEngine:
    def __init__(self):
        self.registry = RulesRegistry.RULES
        
    def generate_finding(self, rule_id: str, status: FindingStatus, 
                         extracted: str, required: str, explanation: str,
                         confidence: Confidence = None) -> Finding:
        
        rule_meta = self.registry.get(rule_id, {})
        if confidence is None:
            confidence = Confidence()
            
        return Finding(
            rule_id=rule_id,
            rule_title=rule_meta.get("title", rule_id),
            status=status,
            extracted_value=extracted,
            required_value_or_condition=required,
            explanation=explanation,
            confidence=confidence,
            statutory_reference=rule_meta.get("statutory_ref", "Unknown"),
            remediation_guidance=rule_meta.get("remediation"),
            penalty_reference=rule_meta.get("penalty_reference")
        )

    def evaluate_case(self, case: InspectionCase) -> InspectionCase:
        # 1. Combine all tokens across uploaded images for now
        # (In Phase 4, we'd map this panel by panel)
        all_text = " ".join([t.get('text', '').lower() for t in case.package_metadata.get('raw_tokens', [])])
        
        # Rule 1: MRP
        mrp_match = re.search(r'(mrp|m\.r\.p|max retail).*?(rs\.?|₹)\s*([\d,]+(?:\.\d{1,2})?)', all_text)
        if mrp_match:
            case.findings.append(self.generate_finding(
                "LM_PC_6_1_e", FindingStatus.COMPLIANT,
                mrp_match.group(0), "MRP in correct format", "MRP correctly detected."
            ))
        else:
            case.findings.append(self.generate_finding(
                "LM_PC_6_1_e", FindingStatus.SEVERE_VIOLATION,
                None, "MRP declaration required", "No valid MRP declaration found in text."
            ))

        # Update overall status based on findings
        has_severe = any(f.status == FindingStatus.SEVERE_VIOLATION for f in case.findings)
        has_minor = any(f.status == FindingStatus.MINOR_INFRACTION for f in case.findings)
        has_review = any(f.status == FindingStatus.REQUIRES_MANUAL_REVIEW for f in case.findings)

        if has_severe:
            case.preliminary_status = FindingStatus.SEVERE_VIOLATION
        elif has_minor:
            case.preliminary_status = FindingStatus.MINOR_INFRACTION
        elif has_review:
            case.preliminary_status = FindingStatus.REQUIRES_MANUAL_REVIEW
        else:
            case.preliminary_status = FindingStatus.COMPLIANT

        case.updated_at = datetime.utcnow()
        return case
