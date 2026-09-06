# Engineering Baseline

| Area | Required evidence |
|---|---|
| **Active app** | `python web_dashboard.py` running Flask on port 5000 |
| **OCR** | `metriguard/cv/ocr.py` using RapidOCR with CLAHE multi-pass preprocessing. Tokens map `text`, `confidence`, and `bbox` (bounding box) to `EvidenceCrop` schemas. |
| **Parser** | *Critical Vulnerability Identified*: Current `rules/engine.py` simply joins all raw tokens into a single flat string (`" ".join(...)`) to regex match MRP and Net Weight. It completely destroys bounding box spatial relationships, meaning it is prone to borrowing values from unrelated rows. |
| **Rules** | `metriguard/rules/engine.py` (CanonicalRulesEngine). `legal_rules.py` and `rules_engine.py` are deprecated but still exist in root. |
| **UI** | Rendered via Flask templates in `web_dashboard.py`. Visual evidence crops are mapped to findings, but UI wording must be audited to avoid "Fake legal certainty". |
| **Storage** | Images stored temporarily in memory (base64). Users stored in `users.json`. Reports generated dynamically, no persistent DB. |
| **Security** | `bcrypt` implemented for password hashing. Missing strict authorization tokens on `POST /api/analyze`. |
| **Tests** | `tests/test_engine.py` running via `pytest`. Covers basic MRP string matching and e-commerce parity. **Gap:** Zero spatial bounding box tests or nutrition table tests. |
| **Claims** | "Preserves spatial relationship" - *False* in current iteration (relies on flat text concatenation). |
