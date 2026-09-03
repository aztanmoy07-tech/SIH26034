from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from enum import Enum
import uuid

class EntityTag(str, Enum):
    B_MANUFACTURER = "B-MANUFACTURER"
    B_NET_QTY = "B-NET_QTY"
    B_MRP = "B-MRP"
    B_MFG_DATE = "B-MFG_DATE"
    B_CONSUMER_CARE = "B-CONSUMER_CARE"
    B_GENERIC_NAME = "B-GENERIC_NAME"
    B_UNIT_SALE_PRICE = "B-UNIT_SALE_PRICE"
    B_FSSAI_LOGO = "B-FSSAI_LOGO"
    B_FSSAI_LIC = "B-FSSAI_LIC"
    B_VEG_MARK = "B-VEG_MARK"
    B_NON_VEG_MARK = "B-NON_VEG_MARK"
    O = "O"

class VerdictState(str, Enum):
    COMPLIANT = "COMPLIANT"
    MINOR_VIOLATION = "MINOR_VIOLATION"
    SEVERE_VIOLATION = "SEVERE_VIOLATION"

class ExtractedToken(BaseModel):
    text: str
    entity_tag: EntityTag
    confidence: float
    bbox_height_px: int
    bbox_width_px: int
    # Clearances required for Schedule II rule 3.3
    clearance_top_px: int = 0
    clearance_bottom_px: int = 0
    clearance_left_px: int = 0
    clearance_right_px: int = 0

class PackageDimensions(BaseModel):
    shape: Literal["rectangular", "cylindrical", "irregular"]
    height_cm: float = 0.0
    width_cm: float = 0.0
    circumference_cm: float = 0.0
    surface_area_cm2: float = 0.0

class AiPayload(BaseModel):
    """
    Mocked payload representing output from PaddleOCR and LayoutLMv3
    """
    scan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tokens: List[ExtractedToken]
    dimensions: PackageDimensions
    px_to_mm_ratio: float = 0.1  # Mock calibration from ArUco marker
    
    # E-commerce parity data (Optional)
    digital_net_weight: Optional[str] = None
    digital_country_of_origin: Optional[str] = None

class Violation(BaseModel):
    rule_section: str
    description: str
    severity: VerdictState

class InspectionReport(BaseModel):
    scan_id: str
    verdict: VerdictState
    violations: List[Violation]
    action_required: str
    pdp_area_cm2: float
