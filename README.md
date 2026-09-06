# MetriGuard — Explainable Legal Metrology Compliance Assistant

**SIH26034 Prototype | Government of India - Department of Consumer Affairs**

MetriGuard is a decision-support system that helps Legal Metrology officers and packaging teams move from package photographs to a defensible preliminary compliance case in under two minutes. 

*MetriGuard does not replace the authorized officer; it gives every officer faster evidence, consistent rules, and a defensible digital trail.*

## 📖 Architecture & Features

This repository implements a canonical, unified Python domain architecture:
- `metriguard/domain/`: Shared schemas (Pydantic) for `InspectionCase`, `PanelType`, and `Finding`.
- `metriguard/cv/`: Computer Vision pipelines for Image Quality gating (Blur/Glare) and multi-pass OCR evidence extraction.
- `metriguard/rules/`: A deterministic rules registry implementing Legal Metrology (PC) Rules, 2011 and FSSAI 2020.
- `metriguard/reports/`: Official export generator (PDF/HTML/CSV/JSON).

### Core Features
- **4-Step Inspector Workspace:** Capture (multi-panel), AI Extract, Human Verify, and Issue Report.
- **Explainable Findings:** Every violation is linked to a visual crop (bounding box) and a statutory reference.
- **E-Commerce Parity:** Compares physical package declarations against digital listings (Net Weight, MRP).
- **Secure Authentication:** `bcrypt` password hashing and protected endpoints.
- **Audit Trails:** Export cases as an immutable HTML/PDF report.

## 🚀 Setup Instructions

1. **Clone and Install**
   ```bash
   git clone https://github.com/aztanmoy07-tech/SIH26034.git
   cd SIH26034
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   python web_dashboard.py
   ```
   Open `http://localhost:5000` in your browser.

3. **Run the Automated Tests**
   ```bash
   pytest tests/ -v
   ```

## 🎥 3-Minute SIH Demo Script

**0:00–0:20 (The Problem):** Explain the bottleneck. "Checking a package manually takes 15 minutes. It's subjective. MetriGuard cuts this to 2 minutes."
**0:20–0:50 (Capture):** Start a new case. Click "Load Test Sample". Show how the UI requires multiple panels (Front, Back, Side). Explain the image quality pre-check for blur/glare.
**0:50–1:20 (Extract & Verify):** Click Extract. Show the AI masking. Move to Step 3 (Verify). Select a "Severe Violation" (e.g., missing PIN code). Show how the AI provides the exact text crop, the required value, and the legal rule (Rule 6(1)(a)).
**1:20–1:50 (Human-in-the-Loop):** Show the "Override as Compliant" button. Emphasize: *MetriGuard is a decision-support tool, the officer retains final authority.*
**1:50–2:15 (E-Commerce Parity):** Show how a digital listing claiming "500g" but a physical label claiming "400g" flags an automated parity violation.
**2:40–3:00 (Export):** Move to Step 4. Click "Export Official PDF". Show the generated report with the legal disclaimer.

## 🔒 Privacy & Data Retention
All uploaded images are processed locally by the CV pipeline. Cases remain in the preliminary state until officially exported. Unauthenticated endpoints are strictly rate-limited.

*Note: The prototype includes sample placeholder contact numbers and emails. Do not use them for real enforcement.*