# MetriGuard: AI-Powered Legal Metrology & FSSAI Compliance Auditor

## 1. Technical Approach
MetriGuard utilizes a hybrid Computer Vision (CV) and Natural Language Processing (NLP) architecture to automate packaging audits. 

**Computer Vision Pipeline:**
* **Multi-Pass OCR:** We implemented a custom 4-pass Optical Character Recognition pipeline using **RapidOCR (ONNX)**. To counter blurry, skewed, or highly reflective packaging, the image undergoes four parallel OpenCV transformations:
  1. CLAHE + Sharpening (for dense text)
  2. Adaptive Threshold Binarization (for tabular data)
  3. Upscaling (for low-resolution scans)
  4. Color Inversion (for light text on dark backgrounds)
* **Deduplication:** Tokens from all passes are merged and deduplicated using spatial Intersection over Union (IoU) to maximize text recovery without duplicate noise.

**Smart Rules Engine (NLP):**
* **Spatial Proximity Detection:** Instead of relying on flawed string concatenation, our engine evaluates OCR tokens dynamically based on their bounding box coordinates. It matches nutrient labels (e.g., "Protein") and searches for numeric values dynamically within a `1.2x` height threshold of the same row, handling highly skewed or multi-column tables perfectly.
* **Statutory Mapping:** Parsed data is fed into a rules engine hardcoded with FSSAI and PC Rules, generating a pass/fail matrix and calculating required minimum font sizes based on the Principal Display Panel (PDP) area.

**Tech Stack:** Python, Flask, OpenCV, RapidOCR (ONNX), PyTorch (YOLOv8), Tailwind CSS, Vanilla JS.

---

## 2. Working Prototype
The MetriGuard prototype is currently **fully functional** and deployed locally. 
* **Capabilities:** Users can upload a package image via the web dashboard. The system instantly classifies the image (e.g., Front Principal Display Panel vs. Nutritional Back Panel).
* **Output:** It generates a real-time compliance report with interactive badges (🔴 SEVERE VIOLATION, 🟡 MINOR INFRACTION, 🟢 COMPLIANT, 🔵 VERIFY MANUALLY).
* **Detailed Breakdown:** For every check (e.g., MRP, Net Quantity, Trans Fats, Allergens), the prototype displays exactly what was extracted from the label, the legal requirement, the statutory reference, and the specific penalty for non-compliance.

---

## 3. Feasibility
**Technical Feasibility:**
* **Hardware Efficiency:** By leveraging ONNX runtime and RapidOCR instead of massive LLMs, the entire auditing engine runs efficiently on standard CPUs without requiring expensive cloud GPUs.
* **Resilience:** The dynamic spatial mapping and custom token boundaries (which account for OCR merging text and numbers like `Protein3.4g`) make the system highly robust against poor-quality real-world images.

**Operational Feasibility:**
* The system is packaged as a lightweight Flask web application, meaning it can be easily containerized (Docker) and deployed to edge devices, mobile inspection apps, or internal corporate networks with minimal friction.

---

## 4. Viability
**Market Viability:**
* **Target Audience:** FMCG manufacturers, packaging design agencies, quality assurance teams, and government Legal Metrology inspectors.
* **Cost vs. Benefit:** Manual audits are time-consuming, prone to human error, and require extensive legal training. MetriGuard reduces audit times from hours to seconds, vastly undercutting the cost of traditional compliance checks.
* **Commercialization:** The engine can be licensed as a B2B SaaS platform, integrated into supply-chain ERPs via API, or provided as a mobile app for on-the-ground regulatory inspectors.

---

## 5. Impact and Benefits
**For Regulators (Government):**
* **Scale & Efficiency:** Empowers inspectors to process hundreds of packages a day simply by taking photos, drastically increasing enforcement of the Legal Metrology Act.

**For Businesses (FMCG & Manufacturers):**
* **Risk Mitigation:** Prevents costly legal penalties. (e.g., Missing Trans Fat declarations or MRP formatting errors can result in fines up to ₹3,00,000 under the FSS Act, or prosecution under the Jan Vishwas Act).
* **Faster Time-to-Market:** Speeds up the packaging approval pipeline by providing instantaneous automated feedback to design teams before printing begins.

**For Consumers:**
* **Safety & Transparency:** Ensures mandatory safety warnings (like bolded Allergen advisories) and fair pricing (MRP with "inclusive of all taxes") are strictly enforced, protecting public health and consumer rights.

---

## 6. Research and References
Our algorithmic rules and compliance logic are strictly referenced from current Indian law:

1. **FSSAI Labelling Regulations:** *Food Safety and Standards (Labelling and Display) Regulations, 2020.* (Used for Nutritional Table structures, Trans Fat mandates, Allergen advisories, and %RDA footnotes).
2. **Legal Metrology (PC) Rules:** *The Legal Metrology (Packaged Commodities) Rules, 2011* (including the 2022 amendments). (Used for Manufacturer PIN code mandates, MRP formatting, Generic Names, and Date formats).
3. **Legal Metrology Act:** *The Legal Metrology Act, 2009.* (Used for baseline packaging definitions and penalty calculations).
4. **Jan Vishwas Act:** *Jan Vishwas (Amendment of Provisions) Act, 2023.* (Used to calculate updated, decriminalized monetary penalties for packaging infractions).
5. **Computer Vision Research:** *RapidOCR & ONNX Runtime Documentation* for efficient multi-language text extraction strategies.
