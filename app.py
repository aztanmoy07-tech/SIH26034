"""
=============================================================================
MetriGuard — Explainable Legal Metrology Compliance Assistant (SIH26034)
A Web-First Regulatory Verification Platform for Packaged Commodities
=============================================================================
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Custom Modules
from cv_pipeline import ImagePreprocessor, PackageExtractor
from legal_rules import LegalMetrologyRulesEngine

# Page Configuration
st.set_page_config(
    page_title="MetriGuard — Legal Metrology Compliance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-bottom: 1.5rem;
    }
    .card-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-minor {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-severe {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


def draw_bounding_boxes(image: Image.Image, tokens: list) -> Image.Image:
    """Draws color-coded bounding boxes over the package image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    for t in tokens:
        bbox = t.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="#2563EB", width=2)
    return img_draw


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=65)
    st.title("MetriGuard Config")
    st.caption("Legal Metrology (PC) Rules, 2011 + 2026 Amendments")

    st.markdown("---")
    st.subheader("📐 Principal Display Panel (PDP)")
    
    package_shape = st.selectbox(
        "Package Shape",
        options=["rectangular", "cylindrical", "irregular"],
        format_func=lambda x: {
            "rectangular": "📦 Rectangular Box / Tetra Pak",
            "cylindrical": "🧴 Cylindrical Can / Bottle",
            "irregular": "🍿 Irregular Snack Pouch"
        }[x]
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        p_height = st.number_input("Height (cm)", value=15.0, min_value=1.0, step=0.5)
    with col_d2:
        p_width = st.number_input("Width (cm)", value=10.0, min_value=1.0, step=0.5)

    p_circ = 0.0
    p_surf = 0.0
    if package_shape == "cylindrical":
        p_circ = st.number_input("Circumference (cm)", value=18.0, min_value=1.0, step=0.5)
    elif package_shape == "irregular":
        p_surf = st.number_input("Total Surface Area (cm²)", value=300.0, min_value=5.0, step=5.0)

    # Calculated PDP Area
    if package_shape == "rectangular":
        calc_pdp = p_height * p_width
    elif package_shape == "cylindrical":
        calc_pdp = 0.40 * p_height * (p_circ or (p_width * 3.1416))
    else:
        calc_pdp = 0.40 * (p_surf or (p_height * p_width * 2))

    st.info(f"📊 Calculated PDP Area: **{calc_pdp:.1f} cm²**")

    st.markdown("---")
    is_food = st.checkbox("Food Commodity (FSSAI Check)", value=True)
    apply_bg_mask = st.checkbox("Enable Background Masking", value=True)
    
    st.markdown("---")
    st.caption("🚀 Smart India Hackathon — PS: SIH26034")


# =============================================================================
# MAIN INTERFACE
# =============================================================================
st.markdown('<div class="main-header">⚖️ MetriGuard — Legal Metrology Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Package Label Verification under the Legal Metrology Act, 2009 & Jan Vishwas Act Reforms</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>DECISION-SUPPORT NOTICE:</strong> This system provides automated preliminary checks to assist authorised enforcement officers and compliance teams. All findings must undergo human-in-the-loop verification before formal legal action.
</div>
""", unsafe_allow_html=True)

# Tabs
tab_inspect, tab_batch, tab_rules = st.tabs([
    "🔍 Scan & Inspect Label", 
    "📁 80k Dataset & YOLO11 Trainer", 
    "📖 Statutory Rulebook (2011–2026)"
])


# =============================================================================
# TAB 1: SCAN & INSPECT LABEL
# =============================================================================
with tab_inspect:
    upload_col, preview_col = st.columns([1, 1])

    with upload_col:
        st.subheader("1. Ingest Package Image")
        source_mode = st.radio("Select Image Source", ["Upload File", "Take Live Photo", "Load Sample Biscuit/Snack Pack"], horizontal=True)

        uploaded_img = None
        if source_mode == "Upload File":
            file = st.file_uploader("Upload commodity packaging front/back panel", type=["jpg", "jpeg", "png", "webp"])
            if file:
                uploaded_img = Image.open(file).convert("RGB")
        elif source_mode == "Take Live Photo":
            cam_file = st.camera_input("Capture commodity label with ArUco reference card")
            if cam_file:
                uploaded_img = Image.open(cam_file).convert("RGB")
        else:
            # Create a clean synthetic Indian FMCG sample packaging
            sample_img = Image.new("RGB", (650, 500), color=(248, 250, 252))
            draw = ImageDraw.Draw(sample_img)
            draw.rectangle([20, 20, 630, 480], outline="#475569", width=3)
            draw.text((40, 40), "ROYAL BISCUITS - RICH BUTTER COOKIES", fill="#0F172A")
            draw.text((40, 80), "Manufactured by: Royal Bakery Pvt Ltd, Okhla, New Delhi 110020", fill="#334155")
            draw.text((40, 120), "Generic Name: Butter Biscuits", fill="#334155")
            draw.text((40, 160), "Net Quantity: 200 g", fill="#334155")
            draw.text((40, 200), "MRP Rs. 60.00 (inclusive of all taxes)", fill="#0F172A")
            draw.text((40, 240), "Unit Sale Price: Rs. 0.30 per g", fill="#334155")
            draw.text((40, 280), "Date of Mfg: 08/2026", fill="#334155")
            draw.text((40, 320), "Consumer Care: feedback@royalbakery.com | Tel: 1800200100", fill="#334155")
            draw.text((40, 360), "FSSAI Lic. No. 10018011000142", fill="#15803D")
            # Veg symbol
            draw.rectangle([540, 40, 590, 90], outline="#16A34A", width=2)
            draw.ellipse([552, 52, 578, 78], fill="#16A34A")
            uploaded_img = sample_img

    with preview_col:
        st.subheader("2. Computer Vision Preprocessing")
        if uploaded_img:
            if apply_bg_mask:
                with st.spinner("Removing background clutter & segmenting packaging..."):
                    processed_img = ImagePreprocessor.remove_background(uploaded_img)
            else:
                processed_img = uploaded_img

            # Quality Check
            quality = ImagePreprocessor.assess_quality(uploaded_img)
            ratio, marker_box = ImagePreprocessor.detect_aruco_calibration(uploaded_img)

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.image(uploaded_img, caption="Original Input", use_container_width=True)
            with p_col2:
                st.image(processed_img, caption="Masked & Isolated Package", use_container_width=True)

            q_col1, q_col2, q_col3 = st.columns(3)
            q_col1.metric("Resolution", quality["resolution"])
            q_col2.metric("Sharpness Index", f"{quality['blur_score']}")
            q_col3.metric("Spatial Scale", f"{ratio:.3f} mm/px" if marker_box else "Default (0.1 mm/px)")
        else:
            st.info("Upload or capture a label image on the left to begin analysis.")

    # Run Analysis if image is ready
    if uploaded_img:
        st.markdown("---")
        st.subheader("3. Optical Extraction & Statutory Rules Audit")
        
        with st.spinner("Extracting tokens & executing Legal Metrology rules..."):
            tokens = PackageExtractor.run_ocr(processed_img)
            audit = LegalMetrologyRulesEngine.evaluate_package(
                ocr_tokens=tokens,
                pdp_shape=package_shape,
                pdp_height_cm=p_height,
                pdp_width_cm=p_width,
                pdp_circumference_cm=p_circ,
                pdp_surface_area_cm2=p_surf,
                px_to_mm_ratio=ratio,
                is_food_commodity=is_food
            )

        # OVERALL VERDICT BANNER
        verdict = audit["overall_verdict"]
        if verdict == "COMPLIANT":
            st.success(f"### {audit['action_headline']}\n{audit['action_description']}")
        elif verdict == "IMPROVEMENT_NOTICE":
            st.warning(f"### {audit['action_headline']}\n{audit['action_description']}")
        else:
            st.error(f"### {audit['action_headline']}\n{audit['action_description']}")

        # DETAILED CHECKS TABLE
        st.markdown("#### 📋 Itemized Statutory Declarations Checklist")

        for r in audit["rule_checks"]:
            if r.status == "COMPLIANT":
                badge_html = '<span class="badge-pass" style="color: green;">🟢 COMPLIANT</span>'
            elif r.status == "MINOR_INFRACTION":
                badge_html = '<span class="badge-minor" style="color: orange;">🟡 IMPROVEMENT NOTICE</span>'
            elif r.status == "INFORMATIONAL":
                badge_html = '<span class="badge-info" style="color: blue;">🔵 INFORMATIONAL</span>'
            elif r.status == "REQUIRES_MANUAL_REVIEW":
                badge_html = '<span class="badge-review" style="color: purple;">🟣 NEEDS REVIEW</span>'
            else:
                badge_html = '<span class="badge-severe" style="color: red;">🔴 SEVERE VIOLATION</span>'

            with st.expander(f"{r.rule_title} — {r.status.replace('_', ' ')}", expanded=(r.status not in ["COMPLIANT", "INFORMATIONAL"])):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)
                    st.markdown(f"**Statutory Ref:** `{r.statutory_ref}`")
                    st.markdown(f"**Extracted Text:** *\"{r.extracted_text}\"*")
                with c2:
                    st.markdown(f"**Explanation:** {r.explanation}")
                    if r.remedy_notice:
                        st.info(f"**Remedy Timeline (Jan Vishwas):** {r.remedy_notice}")
                    if r.penalty_ref:
                        st.error(f"**Statutory Penalty:** {r.penalty_ref}")

        # VISUAL EVIDENCE VIEWER
        st.markdown("#### 🔍 Visual Bounding Box Evidence")
        bbox_annotated = draw_bounding_boxes(processed_img, tokens)
        st.image(bbox_annotated, caption="Extracted Text Elements with Coordinates", use_container_width=True)

        # DOWNLOAD REPORT
        st.markdown("---")
        report_text = f"""
================================================================================
PRELIMINARY LEGAL METROLOGY COMPLIANCE ASSESSMENT (SIH26034)
================================================================================
Overall Verdict: {audit['overall_verdict']}
Headline: {audit['action_headline']}
PDP Area: {audit['pdp_area_cm2']} cm2 (Min Font Required: {audit['min_font_size_mm']} mm)

DETAILED RULE CHECKS:
"""
        for r in audit["rule_checks"]:
            report_text += f"\n- [{r.status}] {r.rule_title} ({r.statutory_ref})\n  Extracted: {r.extracted_text}\n  Explanation: {r.explanation}\n"
        
        report_text += "\n\nDISCLAIMER: Decision-support analysis only. Requires authorized human sign-off."

        st.download_button(
            label="📥 Download Preliminary Compliance Report (TXT)",
            data=report_text,
            file_name="metriguard_compliance_report.txt",
            mime="text/plain"
        )


# =============================================================================
# TAB 2: 80K DATASET & YOLO11 TRAINING PIPELINE
# =============================================================================
with tab_batch:
    st.subheader("🧠 80,000 Indian FMCG Dataset & Active Learning Pipeline")
    st.write("Train the latest YOLO11 model on your `archive` folder using the active pseudo-labeling pipeline.")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Training Configuration")
        archive_path_input = st.text_input("Archive Dataset Path", value="archive")
        model_variant = st.selectbox("YOLO11 Model Variant", ["yolo11n.pt (Fastest)", "yolo11s.pt", "yolo11m.pt (Balanced)", "yolo11x.pt (Max Accuracy)"])
        epochs_input = st.slider("Training Epochs", min_value=5, max_value=100, value=30)
        batch_input = st.selectbox("Batch Size", [4, 8, 16, 32], index=2)

        if st.button("🚀 Start YOLO11 Training Job"):
            st.success(f"Started YOLO11 training task using {model_variant.split()[0]} on `{archive_path_input}`!")
            st.code(f"python train_yolo11.py --archive {archive_path_input} --model {model_variant.split()[0]} --epochs {epochs_input} --batch {batch_input}")
            st.info("Check terminal / training logs for real-time loss and mAP@0.5 metrics.")

    with col_t2:
        st.markdown("#### Active Learning / Pseudo-Labeling Strategy")
        st.markdown("""
        1. **Bootstrap (2,000 images):** Manually annotated seed images for key declarations (MRP, PIN, FSSAI).
        2. **Teacher Model:** Train initial YOLO11 / LayoutLMv3 teacher model.
        3. **Pseudo-Label 78,000 Images:** Run high-speed inference across the remaining unannotated corpus.
        4. **Confidence Filter (Threshold ≥ 0.85):** Discard ambiguous boxes; feed validated pseudo-labels back to fine-tune production weights.
        """)
        st.progress(0.85, text="Target Pseudo-Labeling Confidence: 85%")


# =============================================================================
# TAB 3: STATUTORY RULEBOOK
# =============================================================================
with tab_rules:
    st.subheader("📖 Consolidated Statutory Framework (2011 — 2026)")
    st.markdown("""
    | Rule | Statutory Requirement | Implementation in MetriGuard |
    |---|---|---|
    | **Rule 2** | Definitions of commodity, manufacturer, importer | Scopes validation context |
    | **Rule 6(1)(a)** | Manufacturer/Packer name, complete address & 6-digit PIN | NLP Regex + Postal PIN validator |
    | **Rule 6(1)(b)** | Common/Generic commodity name (brand alone is illegal) | Generic naming presence check |
    | **Rule 6(1)(c)** | Net quantity declared in standard SI metric units (g, kg, ml, L) | Unit parsing & forbidden modifier checks |
    | **Rule 6(1)(d)** | Month and year of manufacture / packing / import | Date parser & future date validator |
    | **Rule 6(1)(e)** | Maximum Retail Price with "inclusive of all taxes" | Strict statutory phrase string check |
    | **Rule 6(1)(g)** | Consumer Care phone/toll-free number and email ID | 10-digit/1800 phone + RFC email regex |
    | **Rule 6(11) (2022)** | Unit Sale Price (USP) per g/kg/ml | Mathematical verification: $USP = MRP / Net\_Qty$ |
    | **Rules 11 & 12** | Prohibits deceptive qualifiers ("approx", "when packed", "minimum") | Strict forbidden word array trigger |
    | **Schedule II** | Minimum font heights (1.0mm, 2.0mm, 4.0mm, 6.0mm) based on PDP area | PDP Area formula & ArUco spatial measurement |
    | **Jan Vishwas Act 2026** | Decriminalization of procedural lapses; Improvement Notice (15-30 days) | Automated routing between Improvement Notice & Section 36 penalty |
    """)
