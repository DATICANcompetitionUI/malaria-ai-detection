import re
import textwrap

# Load the current file code
with open("app/streamlit_app.py", "r") as f:
    code = f.read()

# -------------------------------------------------------------
# STEP 1: Add render_dashboard() helper function
# -------------------------------------------------------------
dashboard_func = """
# PART 3
def render_dashboard():
    st.markdown(\"\"\"
    <div style="background: linear-gradient(135deg, #0d0d1a, #1a1a2e, #0f3460);
                border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;
                border: 1px solid rgba(100,255,218,0.1);">
        <h2 style="color: #e94560; margin-bottom: 0.3rem;">PlasmoID AI</h2>
        <p style="color: #a8b2d1; margin: 0;">
            AI Clinical Assistant for Malaria Microscopy
        </p>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    st.markdown("### 📋 Quick Facts")
    fact_col1, fact_col2 = st.columns(2)
    with fact_col1:
        st.markdown(\"\"\"
        | | |
        |---|---|
        | **Model** | YOLOv8n |
        | **Dataset** | BBBC041 |
        | **Species** | *P. vivax* |
        | **mAP@0.5** | 63.1% |
        | **Inference** | ~140ms/img (CPU) |
        \"\"\")
    with fact_col2:
        st.markdown(\"\"\"
        | | |
        |---|---|
        | **Detection** | 5-class object detection |
        | **WHO Classification** | Enabled |
        | **Human Verification** | Enabled |
        | **Clinical Reports** | PDF + CSV |
        | **Slide Quality Check** | Enabled |
        \"\"\")
    
    st.markdown("### 🔄 Clinical Workflow")
    st.markdown(\"\"\"
    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; 
                align-items: center; font-size: 0.9rem; color: #a8b2d1;">
        <span>🧫 Upload</span> <span style="color:#64ffda;">→</span>
        <span>🔍 Quality Check</span> <span style="color:#64ffda;">→</span>
        <span>🧠 Detection</span> <span style="color:#64ffda;">→</span>
        <span>👨⚕️ Verification</span> <span style="color:#64ffda;">→</span>
        <span>📈 Parasitemia</span> <span style="color:#64ffda;">→</span>
        <span>🏥 WHO Classification</span> <span style="color:#64ffda;">→</span>
        <span>📄 Report</span>
    </div>
    \"\"\", unsafe_allow_html=True)
    
    st.markdown("### ✓ Key Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("✓ Human-in-the-loop AI")
        st.markdown("✓ Explainable Detection")
    with feat_col2:
        st.markdown("✓ Clinical PDF Report")
        st.markdown("✓ Slide Quality Check")
    with feat_col3:
        st.markdown("✓ Batch Processing")
        st.markdown("✓ CPU-Only Inference")
    
    st.markdown("---")
    st.caption(
        "⚠️ For research and decision support. Not intended as a "
        "standalone diagnostic tool. All findings require confirmation "
        "by a qualified microscopist."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([1, 1, 1])
    with cta_col:
        if st.button("🧫 Start New Diagnosis", use_container_width=True, 
                     type="primary"):
            st.session_state["current_page"] = "diagnosis"
            st.rerun()
"""

# Let's insert render_dashboard() right before `def main():`
code = code.replace("def main():", dashboard_func + "\n\ndef main():")

# -------------------------------------------------------------
# STEP 2: Initialize Session States for current_page, sliders, etc.
# -------------------------------------------------------------
session_state_inits = """    # PART 2a
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "dashboard"
    if "weights_path" not in st.session_state:
        st.session_state["weights_path"] = "models/best.pt"
    if "conf_threshold" not in st.session_state:
        st.session_state["conf_threshold"] = 0.20
    if "iou_threshold" not in st.session_state:
        st.session_state["iou_threshold"] = 0.45
    if "show_rbc" not in st.session_state:
        st.session_state["show_rbc"] = False
"""

# Insert these session state initialisations right after `st.session_state.dark_mode = False`
target_session_state_end = '    if "dark_mode" not in st.session_state:\n        st.session_state.dark_mode = False'
code = code.replace(target_session_state_end, target_session_state_end + "\n" + session_state_inits)

# Retrieve variables at the top of main()
variables_retrieval = """
    # Local variables for model settings retrieved from session state
    weights_path = st.session_state["weights_path"]
    conf_threshold = st.session_state["conf_threshold"]
    iou_threshold = st.session_state["iou_threshold"]
    show_rbc = st.session_state["show_rbc"]
"""
# Insert after session generated/loaded properties or right before css block
code = code.replace('    # CHANGE — CSS Polish', variables_retrieval + '\n    # CHANGE — CSS Polish')


# -------------------------------------------------------------
# STEP 3: Extract the Download Section from Single Image Mode
# -------------------------------------------------------------
# Locate `# --- Download section ---` to the end of the `with dl_col3:` block.
download_section_start = "            # --- Download section ---"
download_section_end = "            # ── FEATURE 2 — Human-in-the-loop verification"

parts = code.split(download_section_start)
if len(parts) < 2:
    print("Error: Could not locate download section start")
    exit(1)

subparts = parts[1].split(download_section_end)
if len(subparts) < 2:
    print("Error: Could not locate download section end")
    exit(1)

download_block_indented = download_section_start + subparts[0]
# Clean/dedent the download block for use on the Reports page
download_block_dedented = textwrap.dedent(download_block_indented)

# Remove download section from single image mode (replace with a blank line or small spacing)
code = code.replace(download_block_indented, "\n            # Download section moved to reports page\n")


# -------------------------------------------------------------
# STEP 4: Replace the Sidebar with Navigation and Metrics
# -------------------------------------------------------------
# Let's locate the entire `with st.sidebar:` block.
sidebar_start = "    with st.sidebar:"
sidebar_end = "    # --- CHANGED: Analysis mode toggle ---"

sidebar_parts = code.split(sidebar_start)
if len(sidebar_parts) < 2:
    print("Error: Could not locate sidebar start")
    exit(1)

sidebar_subparts = sidebar_parts[1].split(sidebar_end)
if len(sidebar_subparts) < 2:
    print("Error: Could not locate sidebar end")
    exit(1)

old_sidebar_block = sidebar_start + sidebar_subparts[0]

# Construct the new sidebar block (Part 2b)
new_sidebar_block = """    # PART 2b
    with st.sidebar:
        st.markdown("## 🔬 MALARI-AI")
        st.caption("Clinical Decision Support")
        st.markdown("---")
        
        nav_options = {
            "dashboard": "🏠 Clinical Dashboard",
            "diagnosis": "🧫 New Diagnosis",
            "reports": "📄 Reports",
            "settings": "⚙️ Settings",
        }
        
        for key, label in nav_options.items():
            is_active = st.session_state["current_page"] == key
            button_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{key}", 
                        use_container_width=True, type=button_type):
                st.session_state["current_page"] = key
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Performance")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("mAP@50", "63.1%", help=f"mAP stand for Mean Average Precision. However, our average precisions per class are: RBC- 99.3%, Ring- 65.5%, Trop- 80.8%, Sch- 35.0%, Gam- 34.8%")
            st.metric("Precision", "58.4%", help="Of all boxes detected, what % were correct.")
        with metrics_col2:
            st.metric("Recall", "67.9%", help="Of all the ground truth boxes, what fraction the model found.")
            st.metric("F1 Score", "62.8%", help="Harmonic mean of precision and recall.")
        st.caption("After full training (epoch 50/50, 5-class).")

        st.markdown("---")
        st.markdown("### About This Tool❔")
        st.markdown(\"\"\"
        | | |
        |---|---|
        | **Model** | YOLOv8n |
        | **Dataset** | BBBC041 |
        | **Species** | *P. vivax* |
        | **Classes** | 5 |
        | **Inference** | ~140ms/img |
        \"\"\")
        st.warning(
            f"Note: Research use only. Not a certified "
            f"medical diagnostic device. Always confirm "
            f"with a qualified microscopist."
        )
"""

code = code.replace(old_sidebar_block, new_sidebar_block)


# -------------------------------------------------------------
# STEP 5: Wrap the Main Content Area into Conditional Page Blocks
# -------------------------------------------------------------
footer_start = "    # CHANGE 2f — Footer team credit"
footer_parts = code.split(footer_start)
if len(footer_parts) < 2:
    print("Error: Could not locate footer credit start")
    exit(1)

content_and_header = footer_parts[0]
content_parts = content_and_header.split("# --- Header ---")
if len(content_parts) < 2:
    print("Error: Could not locate header start")
    exit(1)

pre_header = content_parts[0]
main_content_flow = "# --- Header ---" + content_parts[1]

# Indent the main_content_flow by 4 spaces
indented_diagnosis_flow = ""
for line in main_content_flow.splitlines():
    if line.strip() == "":
        indented_diagnosis_flow += "\n"
    else:
        indented_diagnosis_flow += "    " + line + "\n"

# Escape curly braces for st.caption format inside f-string
conditional_routing_block = f"""
    # PART 2c
    if st.session_state["current_page"] == "dashboard":
        render_dashboard()
        
    elif st.session_state["current_page"] == "diagnosis":
{indented_diagnosis_flow}
    elif st.session_state["current_page"] == "reports":
        st.markdown("### 📄 Reports & Downloads")
        st.caption("Download reports and raw data for completed analyses")
        st.markdown("---")
        
        result = st.session_state.get("current_result")
        if result is None:
            st.info("No analysis yet. Go to New Diagnosis to get started.")
        else:
            image_name = st.session_state.get("current_image_name", "image.png")
            image_bgr = st.session_state.get("current_image_bgr")
            uncertain_count, _ = _count_tiers(result)
            
            # Replaced with the extracted download block
{textwrap.indent(download_block_dedented, '        ')}
            
    elif st.session_state["current_page"] == "settings":
        st.markdown("### ⚙️ Settings")
        st.caption("Adjust AI model thresholds and configuration")
        st.markdown("---")
        
        st.session_state["conf_threshold"] = st.slider(
            "Detection Sensitivity",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state["conf_threshold"],
            step=0.05,
            help="Minimum detection confidence. Lower = more detections (more false positives).",
        )

        st.session_state["iou_threshold"] = st.slider(
            "NMS Overlap Tolerance",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state["iou_threshold"],
            step=0.05,
            help="Non-Maximum Suppression threshold. Lower = fewer overlapping boxes.",
        )

        st.session_state["show_rbc"] = st.checkbox(
            "Show Red Blood Cells",
            value=st.session_state["show_rbc"],
            help="Toggle visibility of healthy RBC detections (reduces visual clutter).",
        )
        
        st.markdown("---")
        with st.expander("⚙️ Advanced / Developer Settings", expanded=True):
            st.session_state["weights_path"] = st.text_input(
                "Model Weights Path",
                value=st.session_state["weights_path"],
                help="Path to trained YOLOv8 .pt file",
            )
            if 'last_quality' in st.session_state:
                q = st.session_state['last_quality']
                st.markdown("**Last slide quality metrics**")
                st.caption(f"Sharpness: {{q['sharpness']}} · "
                           f"Brightness: {{q['brightness']}}/255 · "
                           f"Saturation: {{q['saturation_pct']}}%")
"""

new_code = pre_header + conditional_routing_block + "\n" + footer_start + footer_parts[1]

with open("app/streamlit_app.py", "w") as f:
    f.write(new_code)

print("Refactoring successfully completed.")
