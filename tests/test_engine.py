import pytest
from datetime import datetime
from metriguard.domain.schemas import InspectionCase, FindingStatus
from metriguard.rules.engine import CanonicalRulesEngine
from metriguard.rules.parity import ECommerceParityChecker

def test_canonical_rules_engine_compliant_mrp():
    """Test that a valid MRP is marked as compliant."""
    engine = CanonicalRulesEngine()
    case = InspectionCase(
        user_id="officer_1",
        role="inspector",
        organization="LM_Dept",
        package_metadata={
            "raw_tokens": [
                {"text": "Net Wt 500g"},
                {"text": "MRP Rs. 150.00 (Incl. of all taxes)"}
            ]
        }
    )
    result = engine.evaluate_case(case)
    mrp_finding = next((f for f in result.findings if f.rule_id == "LM_PC_6_1_e"), None)
    
    assert mrp_finding is not None
    assert mrp_finding.status == FindingStatus.COMPLIANT
    assert "150.00" in mrp_finding.extracted_value

def test_canonical_rules_engine_missing_mrp():
    """Test that missing MRP triggers a severe violation."""
    engine = CanonicalRulesEngine()
    case = InspectionCase(
        user_id="officer_1",
        role="inspector",
        organization="LM_Dept",
        package_metadata={
            "raw_tokens": [
                {"text": "Net Wt 500g"},
                {"text": "Best Before 6 months"}
            ]
        }
    )
    result = engine.evaluate_case(case)
    mrp_finding = next((f for f in result.findings if f.rule_id == "LM_PC_6_1_e"), None)
    
    assert mrp_finding is not None
    assert mrp_finding.status == FindingStatus.SEVERE_VIOLATION
    assert result.preliminary_status == FindingStatus.SEVERE_VIOLATION

def test_ecommerce_parity_mismatch():
    """Test that an e-commerce price exceeding physical MRP triggers a violation."""
    case = InspectionCase(
        user_id="officer_1",
        role="inspector",
        organization="LM_Dept",
        package_metadata={
            "extracted_mrp": "150.00"
        }
    )
    digital_listing = {
        "mrp": "180.00" # Exceeds physical
    }
    
    result = ECommerceParityChecker.check_parity(case, digital_listing)
    parity_finding = next((f for f in result.findings if f.rule_id == "ECOM_PARITY_MRP"), None)
    
    assert parity_finding is not None
    assert parity_finding.status == FindingStatus.SEVERE_VIOLATION
    assert "Exceeds Physical MRP" in parity_finding.rule_title

def test_ecommerce_parity_compliant():
    """Test that matching e-commerce parity passes."""
    case = InspectionCase(
        user_id="officer_1",
        role="inspector",
        organization="LM_Dept",
        package_metadata={
            "extracted_mrp": "150.00",
            "extracted_net_weight": "500g"
        }
    )
    digital_listing = {
        "mrp": "150.00",
        "net_weight": "500g"
    }
    
    result = ECommerceParityChecker.check_parity(case, digital_listing)
    assert len(result.findings) == 0
