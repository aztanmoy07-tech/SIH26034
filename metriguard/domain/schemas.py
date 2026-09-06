from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime
import uuid

class PanelType(str, Enum):
    FRONT_PDP = "FRONT_PDP"
    NUTRITIONAL_BACK_PANEL = "NUTRITIONAL_BACK_PANEL"
    INGREDIENT_PANEL = "INGREDIENT_PANEL"
    SIDE_DECLARATION_PANEL = "SIDE_DECLARATION_PANEL"
    BARCODE_QR_PANEL = "BARCODE_QR_PANEL"
    UNKNOWN_OR_INCOMPLETE = "UNKNOWN_OR_INCOMPLETE"

class FindingStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    MINOR_INFRACTION = "MINOR_INFRACTION"
    SEVERE_VIOLATION = "SEVERE_VIOLATION"
    REQUIRES_MANUAL_REVIEW = "REQUIRES_MANUAL_REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INFORMATIONAL = "INFORMATIONAL"

class HumanReviewState(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    OVERRIDDEN = "OVERRIDDEN"
    DISMISSED = "DISMISSED"

class Confidence(BaseModel):
    ocr_confidence: float = 1.0
    panel_classification_confidence: float = 1.0
    entity_extraction_confidence: float = 1.0
    rule_evaluation_confidence: float = 1.0
    overall_finding_confidence: float = 1.0

class EvidenceCrop(BaseModel):
    image_path: str
    bbox: List[int] # [x1, y1, x2, y2]
    description: str

class Finding(BaseModel):
    rule_id: str
    rule_title: str
    status: FindingStatus
    extracted_value: Optional[str] = None
    required_value_or_condition: str
    explanation: str
    confidence: Confidence
    evidence_refs: List[EvidenceCrop] = []
    statutory_reference: str
    source_url: Optional[str] = None
    ruleset_effective_date: Optional[str] = None
    remediation_guidance: Optional[str] = None
    penalty_reference: Optional[str] = None
    human_review_state: HumanReviewState = HumanReviewState.PENDING
    reviewer_note: Optional[str] = None

class InspectionCase(BaseModel):
    inspection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    role: str
    organization: str
    location: Optional[str] = None
    product_name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    batch_lot_number: Optional[str] = None
    package_metadata: Dict = {}
    ruleset_version: str = "LM_PC_RULES_2011_V1"
    required_panels: List[PanelType] = [PanelType.FRONT_PDP, PanelType.NUTRITIONAL_BACK_PANEL]
    uploaded_images: List[str] = []
    findings: List[Finding] = []
    preliminary_status: Optional[FindingStatus] = None
    audit_events: List[str] = []
