# MetriGuard Implementation Baseline

## Current Entry Points and Execution
- **Flask Web Dashboard (`web_dashboard.py`)**: The primary UI. Run via `python web_dashboard.py`. Serves HTML and API endpoints on port 5000.
- **Streamlit Prototype (`app.py`)**: An alternative UI prototype. Run via `streamlit run app.py`.
- **FastAPI Prototype (`main.py`)**: An isolated rules API. Run via `uvicorn main:app`. 

## Current API Routes and UI Flows
**Flask (`web_dashboard.py`)**:
- `POST /api/analyze`: Main endpoint taking base64 images and returning compliance JSON.
- `POST /api/signup`, `POST /api/login`: Authentication endpoints.
- `GET/POST /api/content/<page>`: CMS logic for terms/privacy/contact.
- `GET /api/test-sample/<filename>`: Loads deterministic test images.

**FastAPI (`main.py`)**:
- `POST /api/v1/analyze`: Accepts structured payload (not images).
- `GET /api/v1/reports/{scan_id}`: Retrieves rules report.

## Current Data Models
There is a massive split in data modeling:
- **`models.py`**: Clean Pydantic schemas (`AiPayload`, `ExtractedToken`, `InspectionReport`, `Violation`). Used *only* by the FastAPI prototype.
- **`legal_rules.py` & `web_dashboard.py`**: Uses raw, untyped Python dictionaries and JSON structures. 

## Dashboard Dependencies
The main `web_dashboard.py` imports `cv_pipeline` (for `PackageExtractor`) and `legal_rules` (for `LegalMetrologyRulesEngine`). It entirely ignores `models.py` and `rules_engine.py`.

## Duplicate Rules & Competing Implementations
- **Contradiction**: `legal_rules.py` implements nutritional, FSSAI, and veg/non-veg checks using dicts. `rules_engine.py` implements Manufacturer, PIN code, and PDP size checks using Pydantic models. 
- The Flask app never calls `rules_engine.py`. The FastAPI app never calls `legal_rules.py`. There is no unified canonical rules engine.

## Existing Test Coverage and Known Failures
- **`test_rules.py`**: A manual execution script containing hardcoded Maggi token fixtures. It prints console output. 
- **Failure**: There is no standard test suite (e.g., `pytest`). No assertions are made. The codebase currently has 0% automated test coverage.

## Current Upload Limits and File Types
- The Flask dashboard accepts base64 encoded strings in the `POST /api/analyze` JSON body. There is no strict backend validation of file types (magic numbers) or size limits, relying entirely on Flask's default request size limits and frontend `<input accept="image/*">`.

## Authentication Behavior and Security Risks
- **Storage**: Uses a flat `users.json` file.
- **Hashing**: Uses `hashlib.sha256(password.encode()).hexdigest()`. This is an unsalted, fast hash, making it highly vulnerable to rainbow table attacks.
- **Session Management**: There are no secure HTTP-only cookies, JWTs, or session checks on the backend routes. The UI manages state purely on the client side. `POST /api/analyze` can be hit without authentication.

## Current Report/Export Behavior
- The frontend generates a client-side JSON download (`MetriGuard_Audit_{Date.now()}.json`) using a Blob. 
- There is no PDF export, no CSV export, and no backend report persistence layer.

## Known Claims vs. Code Reality
- *Claim*: Multi-panel package inspection. *Reality*: The system only processes a single image at a time.
- *Claim*: Secure authentication. *Reality*: Unsalted SHA-256 and no route protection.
- *Claim*: Defensible digital trail. *Reality*: No persistence layer (database) exists to store cases permanently.
