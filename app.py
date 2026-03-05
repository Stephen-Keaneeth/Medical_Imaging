"""
app.py
------
MediScan AI — Medical Image Diagnostic Platform
Streamlit front-end.  Run with:  streamlit run app.py
"""

import streamlit as st
from PIL import Image
import torch

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .stApp {
    background: #0a0f1e;
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1428 !important;
    border-right: 1px solid #1e2d4a;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span {
    color: #94a3b8 !important;
  }

  /* ── Header card ── */
  .header-card {
    background: linear-gradient(135deg, #0f2044 0%, #0a1628 60%, #061020 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .header-card::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem 0;
  }
  .header-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
  }
  .header-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 0.8rem;
  }

  /* ── Section cards ── */
  .scan-card {
    background: #0d1830;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
  }
  .card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  /* ── Result card ── */
  .result-card {
    background: linear-gradient(135deg, #051a0f 0%, #061520 100%);
    border: 1px solid #14532d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
  }
  .result-card.high-risk {
    background: linear-gradient(135deg, #1a0505 0%, #1a0a0a 100%);
    border-color: #7f1d1d;
  }
  .result-card.medium-risk {
    background: linear-gradient(135deg, #1a1505 0%, #1a1205 100%);
    border-color: #78350f;
  }

  .diagnosis-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #4ade80;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .diagnosis-label.risk { color: #f87171; }
  .diagnosis-label.warn { color: #fbbf24; }

  .diagnosis-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f0fdf4;
    margin-bottom: 0.3rem;
  }
  .diagnosis-value.risk { color: #fecaca; }
  .diagnosis-value.warn { color: #fef3c7; }

  /* ── Probability bars ── */
  .prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.55rem;
    gap: 10px;
  }
  .prob-name {
    font-size: 0.82rem;
    color: #94a3b8;
    min-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #1e2d4a;
    border-radius: 3px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
  }
  .prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    min-width: 42px;
    text-align: right;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploadDropzone"] {
    background: #0d1830 !important;
    border: 1px dashed #1e3a5f !important;
    border-radius: 10px !important;
    color: #64748b !important;
  }

  /* ── Selectbox ── */
  .stSelectbox > div > div {
    background: #0d1830 !important;
    border: 1px solid #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* ── Warning banner ── */
  .disclaimer-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #92400e;
    color: #fbbf24;
    margin-top: 1.5rem;
  }

  /* ── Info pills ── */
  .info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    color: #7dd3fc;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    font-family: 'Space Mono', monospace;
  }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: #0284c7;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: #0369a1;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(2,132,199,0.3);
  }
  div[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e2d4a;
  }

  /* hide default header */
  header[data-testid="stHeader"] { display: none; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after CSS so Streamlit doesn't interfere) ────────────────────────
from disease_classifier import DiagnosticEngine
from model_loader import get_registered_scan_types, get_model_config
from image_preprocessing import get_scan_description


# ── Cached engine (one per session) ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine() -> DiagnosticEngine:
    return DiagnosticEngine(device="auto")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#38bdf8;font-weight:700;">
        🔬 MediScan AI
      </div>
      <div style="color:#475569;font-size:0.75rem;margin-top:4px;">Diagnostic Platform v0.1</div>
    </div>
    """, unsafe_allow_html=True)

    scan_types = get_registered_scan_types()
    selected_scan = st.selectbox(
        "Select Scan Type",
        options=scan_types,
        index=0,
        help="Choose the type of medical scan you are uploading.",
    )

    cfg = get_model_config(selected_scan)

    st.markdown("---")
    st.markdown(f"""
    <div class="card-label">Model Info</div>
    <div style="margin-bottom:0.5rem">
      <span class="info-pill">⚙ {cfg.backbone}</span>
      <span class="info-pill">🏷 {cfg.num_classes} classes</span>
    </div>
    <div style="color:#475569;font-size:0.78rem;line-height:1.5;">{cfg.description}</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div class="card-label">Detectable Conditions</div>
    """, unsafe_allow_html=True)
    for cls in cfg.class_names:
        icon = "✅" if cls.lower().startswith("normal") else "🔹"
        st.markdown(f"<div style='color:#94a3b8;font-size:0.8rem;margin:3px 0;'>{icon} {cls}</div>", unsafe_allow_html=True)

    device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown("---")
    st.markdown(f"<div style='color:#334155;font-size:0.72rem;font-family:Space Mono,monospace;'>Device: {device_label}</div>", unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
  <div class="header-title">🔬 MediScan AI</div>
  <p class="header-subtitle">AI-powered modular medical image diagnostic platform</p>
  <div class="header-badge">PROTOTYPE · HACKATHON BUILD · NOT FOR CLINICAL USE</div>
</div>
""", unsafe_allow_html=True)


col_upload, col_result = st.columns([1, 1], gap="large")

# ── Upload column ─────────────────────────────────────────────────────────────
with col_upload:
    st.markdown('<div class="card-label">Upload Medical Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "dcm"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)

        st.markdown(f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:0.5rem;">
          <span class="info-pill">📐 {image.width} × {image.height}px</span>
          <span class="info-pill">🎨 {image.mode}</span>
          <span class="info-pill">📁 {uploaded_file.size // 1024} KB</span>
        </div>
        """, unsafe_allow_html=True)

        scan_desc = get_scan_description(selected_scan)
        st.markdown(f"""
        <div style="margin-top:0.8rem;padding:0.7rem 1rem;background:#0d1830;border:1px solid #1e2d4a;border-radius:8px;">
          <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#38bdf8;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">Preprocessing</div>
          <div style="color:#64748b;font-size:0.78rem;">{scan_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#0d1830;border:1px dashed #1e3a5f;border-radius:12px;
                    padding:3rem 1rem;text-align:center;margin-top:0.5rem;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">🩻</div>
          <div style="color:#334155;font-size:0.85rem;">No image uploaded yet</div>
          <div style="color:#1e2d4a;font-size:0.75rem;margin-top:4px;">
            Supports PNG, JPEG, BMP, TIFF
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Run button (full width, between cols visually) ────────────────────────────
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 1])
with run_col:
    run_btn = st.button("▶  Run Diagnostic", disabled=(uploaded_file is None))


# ── Result column ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="card-label">Diagnostic Result</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="background:#0d1830;border:1px solid #1e2d4a;border-radius:12px;
                    padding:3rem 1rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:0.5rem;opacity:0.3;">📊</div>
          <div style="color:#1e2d4a;font-size:0.85rem;">
            Upload an image and press Run Diagnostic
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif not run_btn and "last_result" not in st.session_state:
        st.markdown("""
        <div style="background:#0d1830;border:1px solid #1e2d4a;border-radius:12px;
                    padding:3rem 1rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:0.5rem;opacity:0.4;">⚡</div>
          <div style="color:#334155;font-size:0.85rem;">Press Run Diagnostic to analyse</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    if run_btn and uploaded_file:
        with st.spinner("Running inference…"):
            engine = get_engine()
            image  = Image.open(uploaded_file)
            result = engine.predict(image, selected_scan)
            st.session_state["last_result"] = result

    # ── Display result ─────────────────────────────────────────────────────────
    if "last_result" in st.session_state and uploaded_file:
        result = st.session_state["last_result"]
        prob   = result.top_probability

        # Risk colour coding
        if result.top_class.lower().startswith("normal") or prob < 0.45:
            risk_class   = ""
            label_class  = ""
            risk_emoji   = "✅"
            risk_text    = "LOW RISK"
        elif prob >= 0.70:
            risk_class   = "high-risk"
            label_class  = "risk"
            risk_emoji   = "⚠️"
            risk_text    = "FLAGGED"
        else:
            risk_class   = "medium-risk"
            label_class  = "warn"
            risk_emoji   = "🔶"
            risk_text    = "REVIEW SUGGESTED"

        st.markdown(f"""
        <div class="result-card {risk_class}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
            <div>
              <div class="diagnosis-label {label_class}">Primary Diagnosis</div>
              <div class="diagnosis-value {label_class}">{result.top_class}</div>
              <div style="color:#475569;font-size:0.8rem;">{selected_scan}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:1.8rem;">{risk_emoji}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                          color:#{'f87171' if label_class=='risk' else 'fbbf24' if label_class=='warn' else '4ade80'};
                          letter-spacing:0.1em;">{risk_text}</div>
            </div>
          </div>

          <div style="margin-bottom:1rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                        color:#475569;letter-spacing:0.1em;text-transform:uppercase;
                        margin-bottom:0.5rem;">Confidence</div>
            <div style="font-family:'Space Mono',monospace;font-size:2.2rem;
                        color:#f0f9ff;font-weight:700;line-height:1;">
              {prob*100:.1f}<span style="font-size:1rem;color:#475569;">%</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability distribution ──────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:1.2rem;">
          <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                      color:#475569;letter-spacing:0.1em;text-transform:uppercase;
                      margin-bottom:0.8rem;">Full Class Distribution</div>
        """, unsafe_allow_html=True)

        colours = [
            "#38bdf8", "#818cf8", "#a78bfa",
            "#f472b6", "#fb923c", "#34d399"
        ]
        for i, (cls, p) in enumerate(zip(result.all_classes, result.all_probabilities)):
            bar_pct  = int(p * 100)
            colour   = colours[i % len(colours)]
            is_top   = (i == 0)
            name_style = f"color:#e2e8f0;font-weight:600;" if is_top else "color:#64748b;"
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-name" style="{name_style}">{cls}</div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill"
                     style="width:{bar_pct}%;background:{'linear-gradient(90deg,'+colour+','+colour+'aa)' if is_top else '#1e3a5f'};"></div>
              </div>
              <div class="prob-pct" style="{'color:'+colour if is_top else ''}">{p*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Technical details ─────────────────────────────────────────────────
        with st.expander("Technical Details", expanded=False):
            st.markdown(f"""
            | Field | Value |
            |---|---|
            | Backbone | `{result.model_backbone}` |
            | Fine-tuned weights | `{"Yes" if result.using_finetuned_weights else "No (ImageNet pretrain)"}` |
            | Inference device | `{result.device_used}` |
            | Input resolution | `224 × 224` |
            | Classes | `{len(result.all_classes)}` |
            """)


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
  ⚠️  <strong>Research Prototype Only.</strong>  This system uses ImageNet-pretrained weights —
  predictions are <em>not</em> medically validated. This tool is intended for demonstration and
  architecture showcase purposes only. Do not use for clinical decision-making.
  Replace <code>weights_path</code> in <code>model_loader.py</code> with fine-tuned weights before
  any real-world evaluation.
</div>
""", unsafe_allow_html=True)
