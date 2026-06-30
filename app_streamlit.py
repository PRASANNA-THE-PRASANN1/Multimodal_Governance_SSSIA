import os
import sys
import json
import time
import tempfile
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st

# Path configuration
if os.path.isdir("NOTEBOOKS"):
    BASE = str(Path(".").resolve())
elif os.path.isdir("../NOTEBOOKS"):
    BASE = str(Path("..").resolve())
else:
    BASE = str(Path(".").resolve().parent)

sys.path.insert(0, os.path.join(BASE, "AGENTS"))

from decision_agent import DecisionAgent

# Page configuration
st.set_page_config(
    page_title="Agentic AI Governance Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich design aesthetics
st.markdown("""
<style>
    .decision-allow {
        background-color: #2ecc71;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .decision-block {
        background-color: #e74c3c;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .decision-sanitize {
        background-color: #f39c12;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("🛡️ Agentic AI Governance Framework")
st.markdown("""
**Multimodal Prompt Injection Detection for Insurance LLM Pipelines.**
*Grounded in OWASP LLM Top 10 2025, NIST AI RMF, and ISO 27001.*
""")

# Lazy load the DecisionAgent using session state
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Agents and loading fine-tuned RoBERTa model..."):
        st.session_state.agent = DecisionAgent(
            gpu=False,
            models_dir=os.path.join(BASE, "MODELS"),
            rules_path=os.path.join(BASE, "RULES", "governance_rules.json")
        )
    st.success("Governance Agent and LLM Safety Guardrails loaded successfully!")

agent = st.session_state.agent

# Sidebar configuration
st.sidebar.header("🛡️ System Settings")
st.sidebar.markdown("**Engine Version**: `1.0.0` (Active)")
st.sidebar.markdown("**Rules Set**: `governance_rules.json` (10 rules)")

prompt_th = st.sidebar.slider("Prompt Score Threshold (G9)", 0.0, 1.0, 0.55, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("### Compliance Frameworks Mapping")
st.sidebar.markdown("- **OWASP LLM01**: Prompt Injection")
st.sidebar.markdown("- **OWASP LLM06**: Sensitive Data Exposure")
st.sidebar.markdown("- **NIST AI RMF**: GOVERN, MAP, MEASURE")

# Main Content Layout (Input Sandbox and Live Analysis)
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📝 Manual Input Sandbox")
    
    input_mode = st.radio("Input Type", ["Text Document", "Multimodal (Text + Image)"])
    
    text_input = ""
    uploaded_image_path = None
    
    if input_mode == "Text Document":
        text_input = st.text_area("Document Content / Claim Description", height=250, placeholder="Enter text to analyze...")
    else:
        text_input = st.text_area("Document Context / Claim Description", height=120, placeholder="Enter supplement text...")
        uploaded_file = st.file_uploader("Upload Document Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                uploaded_image_path = tmp.name
            st.image(uploaded_file, caption="Uploaded Document Preview", use_column_width=True)

    severity_hint = st.selectbox("Metadata Severity Hint", ["low", "medium", "high", "critical"], index=1)
    
    analyze_btn = st.button("🚀 Run Governance Analysis", type="primary")

# Run analysis and print result
with col_output:
    st.subheader("📊 Live Analysis & Decision")
    
    if analyze_btn:
        if input_mode == "Text Document" and not text_input.strip():
            st.warning("Please provide some text to analyze.")
        elif input_mode == "Multimodal (Text + Image)" and not uploaded_image_path:
            st.warning("Please upload a document image.")
        else:
            with st.spinner("Processing document through agentic pipeline..."):
                t_start = time.perf_counter()
                
                # Override threshold
                agent._prompt_threshold = prompt_th
                
                try:
                    result = agent.process(
                        image_path=uploaded_image_path,
                        text=text_input,
                        severity=severity_hint,
                        sample_id=f"WEB_{int(time.time())}"
                    )
                    
                    t_elapsed = (time.perf_counter() - t_start) * 1000
                    
                    # Display decision banner
                    dec = result.decision
                    if dec == "ALLOW":
                        st.markdown(f'<div class="decision-allow">✅ ALLOW</div>', unsafe_allow_html=True)
                    elif dec == "BLOCK":
                        st.markdown(f'<div class="decision-block">❌ BLOCK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="decision-sanitize">⚠️ SANITIZE</div>', unsafe_allow_html=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{result.risk_score:.4f}</div><div class="metric-label">Risk Score</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{result.prompt_score:.4f}</div><div class="metric-label">Prompt Score</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{result.vision_score:.4f}</div><div class="metric-label">Vision Score</div></div>', unsafe_allow_html=True)
                    
                    # Policy Reason card
                    st.markdown("### 🔍 Policy Decision Trace")
                    st.info(f"**Reason:** {result.reason}")
                    
                    if result.governance_rule_triggered:
                        st.warning(f"**Triggered Rule:** `{result.governance_rule_triggered}` | **Policy Ref:** `{result.policy_ref}`")
                        if result.sanitization_action:
                            st.code(f"Sanitization Action: {result.sanitization_action}")
                            
                    # Show processing times
                    st.markdown(f"*Pipeline execution complete in **{t_elapsed:.2f} ms** (Audit ID: `{result.audit_log_id}`)*")
                    
                    # Show detailed feature breakdown
                    st.markdown("### 🧩 Extracted Features (Feature Engineer)")
                    
                    feats = {
                        "malicious_probability": result.prompt_score,
                        "vision_score": result.vision_score,
                        "ocr_confidence": result.confidence,
                        "metadata_severity": severity_hint
                    }
                    st.json(feats)
                    
                except Exception as e:
                    st.error(f"Pipeline execution error: {e}")
                finally:
                    # Cleanup temp image
                    if uploaded_image_path and os.path.exists(uploaded_image_path):
                        try:
                            os.unlink(uploaded_image_path)
                        except Exception:
                            pass
    else:
        st.write("Submit inputs in the sandbox pane to trigger live evaluation.")

# Historical logs section (Audit Trail)
st.markdown("---")
st.subheader("📜 Recent Auditable Logs History (RESULTS/audit_log.jsonl)")

audit_file_path = os.path.join(os.path.join(BASE, "RESULTS"), "audit_log.jsonl")
if os.path.exists(audit_file_path):
    try:
        with open(audit_file_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if lines:
            lines.reverse()
            log_df = pd.DataFrame(lines).head(5)
            cols = ["timestamp", "sample_id", "decision", "risk_score", "prompt_score", "vision_score", "governance_rule_triggered", "reason"]
            st.dataframe(log_df[[c for c in cols if c in log_df.columns]])
        else:
            st.write("Audit log is empty.")
    except Exception as e:
        st.write(f"Could not load audit log history: {e}")
else:
    st.write("No audit log history file found. Run a manual test or evaluation script first.")
