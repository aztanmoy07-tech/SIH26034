from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict
from models import AiPayload, InspectionReport
from rules_engine import ComplianceService

app = FastAPI(
    title="MetriGuard API (FastAPI)",
    description="Legal Metrology Compliance Checker Backend - SIH26034",
    version="1.0.0"
)

# Mock PostgreSQL database using an in-memory dictionary
inspection_db: Dict[str, InspectionReport] = {}
payload_db: Dict[str, AiPayload] = {}

compliance_service = ComplianceService()

@app.post("/api/v1/analyze", response_model=AiPayload, status_code=status.HTTP_201_CREATED)
async def analyze_image_payload(payload: AiPayload):
    """
    AI Ingestion Endpoint.
    In a real scenario, this would accept an image upload, run the YOLOv11-OBB and 
    LayoutLMv3 models, and generate the extracted tokens and dimensions.
    Here, it accepts the structured AI Payload, runs the deterministic rules engine,
    and stores the results.
    """
    # Store the raw AI payload
    payload_db[payload.scan_id] = payload
    
    # Run the Rules Engine
    report = compliance_service.process_inspection(payload)
    
    # Store the resulting report in the mock DB
    inspection_db[report.scan_id] = report
    
    # Return the payload (simulating output from PaddleOCR/LayoutLMv3 as requested)
    return payload


@app.get("/api/v1/reports/{scan_id}", response_model=InspectionReport)
async def get_report(scan_id: str):
    """
    Reporting Endpoint.
    Generates a structured JSON summary of the violations and the specific 
    sections of the 2011 Act breached, classified by the Jan Vishwas Act rules.
    """
    if scan_id not in inspection_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Scan ID not found. Ensure the image was processed via /api/v1/analyze."
        )
    return inspection_db[scan_id]


@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "message": "MetriGuard Backend Rules Engine Operational"}
