from typing import Dict, Any

from ..domain.schemas import InspectionCase, Finding, FindingStatus, Confidence

class ECommerceParityChecker:
    """
    Phase 8: E-commerce Parity.
    Compares physical packaging evidence against online digital listings.
    """
    
    @classmethod
    def check_parity(cls, case: InspectionCase, digital_listing: Dict[str, Any]) -> InspectionCase:
        """
        Cross-references digital claims (like Net Weight, MRP, Origin) 
        with the physical AI findings.
        """
        if not digital_listing:
            return case # Optional module, skip if no digital data provided

        # Example: Net Weight Parity
        digital_weight = digital_listing.get("net_weight")
        physical_weight = case.package_metadata.get("extracted_net_weight")

        if digital_weight and physical_weight:
            if digital_weight.lower() != physical_weight.lower():
                finding = Finding(
                    rule_id="ECOM_PARITY_NET_QTY",
                    rule_title="Digital vs. Physical Net Quantity Mismatch",
                    status=FindingStatus.SEVERE_VIOLATION,
                    extracted_value=f"Physical: {physical_weight} | Online: {digital_weight}",
                    required_value_or_condition="Online listing must exactly match physical package declarations.",
                    explanation="The net weight advertised on the e-commerce listing contradicts the physical label.",
                    confidence=Confidence(rule_evaluation_confidence=0.99),
                    statutory_reference="Consumer Protection (E-Commerce) Rules, 2020",
                    remediation_guidance="Update the digital listing to match the physical product exactly.",
                )
                case.findings.append(finding)
                case.preliminary_status = FindingStatus.SEVERE_VIOLATION

        # Example: MRP Parity
        digital_mrp = digital_listing.get("mrp")
        physical_mrp = case.package_metadata.get("extracted_mrp")
        
        if digital_mrp and physical_mrp:
            # Strip non-numeric for comparison
            d_val = ''.join(filter(str.isdigit, str(digital_mrp)))
            p_val = ''.join(filter(str.isdigit, str(physical_mrp)))
            
            if d_val != p_val:
                finding = Finding(
                    rule_id="ECOM_PARITY_MRP",
                    rule_title="Digital MRP Exceeds Physical MRP",
                    status=FindingStatus.SEVERE_VIOLATION,
                    extracted_value=f"Physical MRP: {physical_mrp} | Online Price: {digital_mrp}",
                    required_value_or_condition="E-commerce price cannot exceed the physical package MRP.",
                    explanation="The product is being sold or advertised online at a price different from the mandated physical MRP.",
                    confidence=Confidence(rule_evaluation_confidence=0.99),
                    statutory_reference="Legal Metrology (Packaged Commodities) Rules, 2011 & E-Commerce Rules 2020",
                    remediation_guidance="Correct the online pricing to not exceed physical MRP.",
                )
                case.findings.append(finding)
                case.preliminary_status = FindingStatus.SEVERE_VIOLATION

        return case
