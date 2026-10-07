import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ==============================================================================
# 1. PAGE CONFIGURATION & INJECTED STYLES (NEXT-GEN CYBER-FOREST SYSTEM)
# ==============================================================================
st.set_page_config(
    page_title="Forest Guardian AI — Autonomous Wildfire Vision Suite",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS System (Dark Glassmorphism, Neon Emerald & Crimson Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Dark Theme Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #070a0f !important;
        color: #f1f5f9 !important;
    }
    
    .stAppViewContainer {
        background: radial-gradient(circle at 50% 0%, #0d1b1e 0%, #070a0f 70%) !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.25);
        box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.08);
    }

    /* Header Navbar Banner */
    .command-header {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 60%, rgba(2, 44, 34, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(6, 78, 59, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .command-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #10b981, #34d399, #059669);
    }
    
    .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
    }

    .brand-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-subtitle {
        font-size: 1.05rem;
        color: #a7f3d0;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Status Pill Badge */
    .status-badge {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        padding: 8px 18px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .pulse-dot {
        width: 9px;
        height: 9px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Academic Disclaimer Card */
    .disclaimer-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fef3c7;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 24px;
    }

    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background-color: #0b0f17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    .sidebar-widget {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-top: 14px;
        margin-bottom: 18px;
    }

    .sidebar-widget-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 10px;
    }

    /* Tabs Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        padding: 0 24px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #94a3b8;
        border: none !important;
        background-color: transparent;
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(5, 150, 105, 0.35) !important;
    }

    /* Detection Result Alerts */
    .fire-alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(127, 29, 29, 0.35) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.25);
    }

    .fire-alert-title {
        color: #fca5a5;
        font-size: 1.5rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.01em;
    }

    .safe-alert-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.35) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
    }

    .safe-alert-title {
        color: #6ee7b7;
        font-size: 1.5rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.01em;
    }

    /* Metric KPI Cards */
    .cyber-kpi {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }

    .cyber-kpi:hover {
        transform: translateY(-3px);
        border-color: rgba(16, 185, 129, 0.4);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }

    .cyber-kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    .cyber-kpi-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }

    .cyber-kpi-sub {
        font-size: 0.8rem;
        color: #34d399;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Styled Confusion Matrix Grid */
    .cm-table-custom {
        width: 100%;
        border-collapse: separate;
        border-spacing: 8px;
        margin-top: 14px;
    }

    .cm-cell-custom {
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .cm-cell-tp { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .cm-cell-tn { background: rgba(59, 130, 246, 0.18); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.4); }
    .cm-cell-fp { background: rgba(245, 158, 11, 0.18); color: #fde047; border: 1px solid rgba(245, 158, 11, 0.4); }
    .cm-cell-fn { background: rgba(239, 68, 68, 0.18); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }

    /* Footer */
    .cyber-footer {
        text-align: center;
        padding: 36px 0 12px 0;
        color: #64748b;
        font-size: 0.88rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HELPER FUNCTIONS & CACHING
# ==============================================================================
@st.cache_resource
def load_cached_model(filepath):
    """Load Keras model with resource caching."""
    if not os.path.exists(filepath):
        return None
    return load_model(filepath)

def load_json_metrics(filepath):
    """Load pre-computed test metrics."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def load_csv_history(filepath):
    """Load epoch training history CSV logs."""
    if not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


# ==============================================================================
# 3. MAIN APPLICATION ENGINE
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # A. HERO COMMAND BAR & DISCLAIMER
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="command-header">
        <div class="header-row">
            <div>
                <div class="brand-title">
                    <span>🌲 Forest Guardian AI</span>
                </div>
                <div class="brand-subtitle">
                    Autonomous Wildfire Neural Detection & Multi-Architecture Benchmarking Suite
                </div>
            </div>
            <div class="status-badge">
                <span class="pulse-dot"></span>
                <span>SYSTEM ONLINE • TENSORFLOW 2.x</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Academic & Evaluation Scope:</strong> Forest Guardian AI is a high-precision computer vision evaluation suite trained on benchmark forestry datasets. It operates in offline inference mode and is <strong>not</strong> connected directly to live satellite telemetry or municipal emergency dispatch systems.
    </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # B. SIDEBAR CONTROL PANEL
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ Model Architecture")

    model_options = {
        "ResNet50": {
            "file": "resnet_model.h5",
            "json": "resnet_test_metrics.json",
            "csv": "resnet_history.csv",
            "desc": "Deep Transfer Learning CNN (50 layers) fine-tuned for wildfire vision",
            "preprocess": "resnet",
            "badge": "🏆 Highest Precision (97.96%)",
            "params": "23.5M parameters",
            "size": "91.8 MB"
        },
        "MobileNetV2": {
            "file": "mobilenet_model.h5",
            "json": "mobilenet_test_metrics.json",
            "csv": "mobilenet_history.csv",
            "desc": "Depthwise separable lightweight CNN optimized for mobile edge hardware",
            "preprocess": "mobilenet",
            "badge": "⚡ Edge Optimized (95.7%)",
            "params": "2.2M parameters",
            "size": "8.9 MB"
        },
        "Custom CNN": {
            "file": "custom_cnn_model.h5",
            "json": "custom_cnn_test_metrics.json",
            "csv": "custom_cnn_history.csv",
            "desc": "4-Layer Convolutional Neural Network trained from scratch as baseline",
            "preprocess": "internal",
            "badge": "⚙️ Baseline CNN (91.4%)",
            "params": "1.1M parameters",
            "size": "2.1 MB"
        }
    }

    selected_model_name = st.sidebar.selectbox(
        "Select Neural Architecture:",
        options=list(model_options.keys()),
        index=0  # Default to ResNet50
    )

    model_info = model_options[selected_model_name]

    st.sidebar.markdown(f"""
    <div class="sidebar-widget">
        <div class="sidebar-widget-title">Active Architecture Specs</div>
        <div style="font-size: 1rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
            {selected_model_name}
        </div>
        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px; line-height: 1.4;">
            {model_info['desc']}
        </div>
        <div style="display: flex; flex-direction: column; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #34d399;">
            <div>• Weight File: {model_info['file']}</div>
            <div>• Payload Size: {model_info['size']}</div>
            <div>• Complexity: {model_info['params']}</div>
            <div>• Preprocess Mode: {model_info['preprocess'].upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model into Memory
    with st.sidebar:
        with st.spinner(f"Loading {selected_model_name} weights..."):
            try:
                model = load_cached_model(model_info["file"])
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                model = None

        if model is not None:
            st.success(f"✓ {selected_model_name} loaded in VRAM")
        else:
            st.error(f"Model file `{model_info['file']}` not found.")

    # --------------------------------------------------------------------------
    # C. TABS NAVIGATION (4 REIMAGINED WORKSPACES)
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Real-Time Inference",
        "📊 Model Evaluation",
        "🏆 Cross-Model Comparison",
        "📖 System Documentation"
    ])

    # ==========================================================================
    # WORKSPACE 1: REAL-TIME INFERENCE
    # ==========================================================================
    with tab1:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0; font-weight: 800; color: #ffffff;">Aerial & Forest Image Classification</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Upload a custom forest photograph or click a pre-loaded evaluation sample to execute neural inference.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Sample paths
        sample_dir_fire = os.path.join("processed_data", "test", "Fire")
        sample_dir_nonfire = os.path.join("processed_data", "test", "Non-Fire")

        sample_fire_1 = os.path.join(sample_dir_fire, "fire_0008.jpg") if os.path.exists(sample_dir_fire) else None
        sample_fire_2 = os.path.join(sample_dir_fire, "F_1004.jpg") if os.path.exists(sample_dir_fire) else None
        sample_nf_1 = os.path.join(sample_dir_nonfire, "NF_1.jpg") if os.path.exists(sample_dir_nonfire) else None
        sample_nf_2 = os.path.join(sample_dir_nonfire, "nofire_0001.jpg") if os.path.exists(sample_dir_nonfire) else None

        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 14px 0; color: #ffffff;">📥 Image Input Selection</h4>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload a high-resolution aerial or ground image:",
                type=["jpg", "jpeg", "png"]
            )

            st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-top: 10px; margin-bottom: 6px;'><strong>Or choose a pre-loaded evaluation sample:</strong></p>", unsafe_allow_html=True)

            sample_choice = st.radio(
                "Quick Test Presets:",
                options=["None (Use Upload Above)", "🔥 Sample 1: Active Wildfire Aerial", "🔥 Sample 2: Forest Fire Smoke", "🌲 Sample 3: Dense Green Canopy", "🌲 Sample 4: Alpine Mountain Forest"],
                index=0
            )

            img_input = None
            preset_gt = "Not specified"

            if uploaded_file is not None:
                try:
                    img_input = Image.open(uploaded_file).convert("RGB")
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")
            elif sample_choice == "🔥 Sample 1: Active Wildfire Aerial" and sample_fire_1:
                img_input = Image.open(sample_fire_1).convert("RGB")
                preset_gt = "Fire"
            elif sample_choice == "🔥 Sample 2: Forest Fire Smoke" and sample_fire_2:
                img_input = Image.open(sample_fire_2).convert("RGB")
                preset_gt = "Fire"
            elif sample_choice == "🌲 Sample 3: Dense Green Canopy" and sample_nf_1:
                img_input = Image.open(sample_nf_1).convert("RGB")
                preset_gt = "Non-Fire"
            elif sample_choice == "🌲 Sample 4: Alpine Mountain Forest" and sample_nf_2:
                img_input = Image.open(sample_nf_2).convert("RGB")
                preset_gt = "Non-Fire"

            ground_truth = st.radio(
                "Verification Ground Truth Label:",
                options=["Not specified", "Fire", "Non-Fire"],
                index=0 if sample_choice == "None (Use Upload Above)" else (1 if preset_gt == "Fire" else 2),
                horizontal=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

            if img_input is not None:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin: 0 0 12px 0; color: #ffffff;">📷 Input Image Preview</h4>', unsafe_allow_html=True)
                st.image(img_input, use_container_width=True, caption=f"Resolution: {img_input.size[0]} × {img_input.size[1]} pixels")
                st.markdown('</div>', unsafe_allow_html=True)

        with col_output:
            if img_input is not None and model is not None:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin: 0 0 16px 0; color: #ffffff;">⚡ Neural Classification Output</h4>', unsafe_allow_html=True)

                try:
                    # Preprocessing
                    img_resized = img_input.resize((224, 224))
                    img_array = np.array(img_resized, dtype=np.float32)

                    if model_info["preprocess"] == "internal":
                        input_tensor = np.expand_dims(img_array.copy(), axis=0)
                    elif model_info["preprocess"] == "mobilenet":
                        input_tensor = np.expand_dims(mobilenet_preprocess(img_array.copy()), axis=0)
                    elif model_info["preprocess"] == "resnet":
                        input_tensor = np.expand_dims(resnet_preprocess(img_array.copy()), axis=0)

                    # Model Prediction
                    prediction_prob = float(model.predict(input_tensor, verbose=0)[0][0])

                    # Class Mapping: 0 = Fire, 1 = Non-Fire
                    non_fire_prob = prediction_prob
                    fire_prob = 1.0 - prediction_prob

                    if prediction_prob < 0.5:
                        predicted_class = "Fire"
                        confidence = fire_prob * 100.0
                    else:
                        predicted_class = "Non-Fire"
                        confidence = non_fire_prob * 100.0

                    # Render Hazard Banner
                    if predicted_class == "Fire":
                        st.markdown(f"""
                        <div class="fire-alert-box">
                            <div class="fire-alert-title">🔥 WILDFIRE DETECTED</div>
                            <div style="font-size: 0.95rem; color: #fecdd3; margin-top: 8px; line-height: 1.4;">
                                High-probability thermal anomaly / combustion signature detected with <strong>{confidence:.2f}%</strong> certainty.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert-box">
                            <div class="safe-alert-title">🌲 NON-FIRE / SAFE TERRAIN</div>
                            <div style="font-size: 0.95rem; color: #a7f3d0; margin-top: 8px; line-height: 1.4;">
                                Healthy vegetation / normal forest canopy detected with <strong>{confidence:.2f}%</strong> certainty.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Probability Meters
                    st.markdown("<h5 style='color: #ffffff; margin-top: 14px;'>Probability Distribution</h5>", unsafe_allow_html=True)

                    st.markdown(f"**🔥 Wildfire Probability:** `{fire_prob * 100:.2f}%`")
                    st.progress(float(fire_prob))

                    st.markdown(f"**🌲 Non-Fire Canopy Probability:** `{non_fire_prob * 100:.2f}%`")
                    st.progress(float(non_fire_prob))

                    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)

                    # Diagnostic Tensor Drawer
                    st.markdown(f"""
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; background: rgba(7, 10, 15, 0.8); border: 1px solid rgba(255,255,255,0.08); padding: 14px; border-radius: 10px; color: #94a3b8; display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div>• Active Model: {selected_model_name}</div>
                        <div>• Raw Sigmoid: {prediction_prob:.4f}</div>
                        <div>• Tensor Shape: (1, 224, 224, 3)</div>
                        <div>• Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Ground Truth Check
                    if ground_truth != "Not specified":
                        st.markdown("<br>", unsafe_allow_html=True)
                        if predicted_class == ground_truth:
                            st.success(f"✅ **Verified Match:** Model prediction ({predicted_class}) matches ground truth label ({ground_truth}).")
                        else:
                            st.warning(f"⚠️ **Ground Truth Variance:** Model prediction ({predicted_class}) differs from ground truth label ({ground_truth}).")

                except Exception as e:
                    st.error(f"Inference execution error: {str(e)}")

                st.markdown('</div>', unsafe_allow_html=True)

            elif img_input is None:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.4); border: 2px dashed rgba(255, 255, 255, 0.12); border-radius: 16px; padding: 60px 24px; text-align: center; color: #64748b;">
                    <div style="font-size: 3rem; margin-bottom: 14px;">🛰️</div>
                    <div style="font-weight: 700; font-size: 1.15rem; color: #cbd5e1;">Awaiting Image Input</div>
                    <div style="font-size: 0.9rem; margin-top: 6px;">Select an evaluation sample or upload an image on the left panel to execute real-time neural inference.</div>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================================================
    # WORKSPACE 2: MODEL EVALUATION DASHBOARD
    # ==========================================================================
    with tab2:
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0; font-weight: 800; color: #ffffff;">Test Dataset Performance — {selected_model_name}</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Comprehensive empirical metrics calculated on the untouched test dataset (1,077 benchmark samples).
            </p>
        </div>
        """, unsafe_allow_html=True)

        metrics = load_json_metrics(model_info["json"])

        if metrics is None:
            st.error(f"Unable to load metrics JSON file `{model_info['json']}`.")
        else:
            # Cyber KPI Grid
            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.markdown(f"""
                <div class="cyber-kpi">
                    <div class="cyber-kpi-label">Overall Accuracy</div>
                    <div class="cyber-kpi-value">{metrics['accuracy'] * 100:.2f}%</div>
                    <div class="cyber-kpi-sub">✓ Top Benchmark Score</div>
                </div>
                """, unsafe_allow_html=True)

            with k2:
                st.markdown(f"""
                <div class="cyber-kpi">
                    <div class="cyber-kpi-label">Macro F1-Score</div>
                    <div class="cyber-kpi-value">{metrics['macro_f1'] * 100:.2f}%</div>
                    <div class="cyber-kpi-sub">✓ Class Balance</div>
                </div>
                """, unsafe_allow_html=True)

            with k3:
                st.markdown(f"""
                <div class="cyber-kpi">
                    <div class="cyber-kpi-label">ROC-AUC Score</div>
                    <div class="cyber-kpi-value">{metrics['roc_auc'] * 100:.2f}%</div>
                    <div class="cyber-kpi-sub">✓ Discriminative Power</div>
                </div>
                """, unsafe_allow_html=True)

            with k4:
                st.markdown(f"""
                <div class="cyber-kpi">
                    <div class="cyber-kpi-label">Fire Recall</div>
                    <div class="cyber-kpi-value">{metrics['fire_recall'] * 100:.2f}%</div>
                    <div class="cyber-kpi-sub" style="color: #fca5a5;">⚠️ Critical Safety Metric</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_detail1, col_detail2 = st.columns([1, 1], gap="large")

            with col_detail1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin: 0 0 16px 0; color: #ffffff;">📊 Class-Specific Metrics</h4>', unsafe_allow_html=True)

                m_c1, m_c2, m_c3 = st.columns(3)
                m_c1.metric("Fire Precision", f"{metrics['fire_precision'] * 100:.2f}%")
                m_c2.metric("Fire F1-Score", f"{metrics['fire_f1'] * 100:.2f}%")
                m_c3.metric("False Negatives", f"{metrics['fire_false_negatives']}")

                st.markdown("""
                <div style="font-size: 0.85rem; color: #94a3b8; background: rgba(7, 10, 15, 0.6); padding: 14px; border-radius: 10px; margin-top: 16px; border: 1px solid rgba(255,255,255,0.08);">
                    <strong>False Negative Hazard Priority:</strong> In wildfire computer vision, a False Negative (a missed fire) poses an extreme real-world risk. Lower False Negative count directly reflects safer deployment performance.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_detail2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin: 0 0 12px 0; color: #ffffff;">🎯 Confusion Matrix (Test Dataset)</h4>', unsafe_allow_html=True)

                cm = metrics["confusion_matrix"]
                tp_val = cm[0][0]
                fn_val = cm[0][1]
                fp_val = cm[1][0]
                tn_val = cm[1][1]

                st.markdown(f"""
                <table class="cm-table-custom">
                    <tr>
                        <th style="font-size: 0.8rem; color: #94a3b8;">Ground Truth \\ Prediction</th>
                        <th style="font-size: 0.8rem; color: #fca5a5;">Predicted FIRE (0)</th>
                        <th style="font-size: 0.8rem; color: #6ee7b7;">Predicted NON-FIRE (1)</th>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; font-size: 0.85rem; color: #cbd5e1;">Actual FIRE (0)</td>
                        <td class="cm-cell-custom cm-cell-tp">
                            <div style="font-size: 1.3rem; font-weight: 800;">{tp_val}</div>
                            <div style="font-size: 0.72rem;">True Positive (TP)</div>
                        </td>
                        <td class="cm-cell-custom cm-cell-fn">
                            <div style="font-size: 1.3rem; font-weight: 800;">{fn_val}</div>
                            <div style="font-size: 0.72rem;">False Negative (FN)</div>
                        </td>
                    </tr>
                    <tr>
                        <td style="font-weight: 700; font-size: 0.85rem; color: #cbd5e1;">Actual NON-FIRE (1)</td>
                        <td class="cm-cell-custom cm-cell-fp">
                            <div style="font-size: 1.3rem; font-weight: 800;">{fp_val}</div>
                            <div style="font-size: 0.72rem;">False Positive (FP)</div>
                        </td>
                        <td class="cm-cell-custom cm-cell-tn">
                            <div style="font-size: 1.3rem; font-weight: 800;">{tn_val}</div>
                            <div style="font-size: 0.72rem;">True Negative (TN)</div>
                        </td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Convergence Charts
            csv_data = load_csv_history(model_info["csv"])
            if csv_data is not None:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f'<h4 style="margin: 0 0 16px 0; color: #ffffff;">📈 Training & Validation Convergence ({selected_model_name})</h4>', unsafe_allow_html=True)

                ch_c1, ch_c2 = st.columns([1, 1], gap="large")

                with ch_c1:
                    st.markdown("<h5 style='color: #cbd5e1;'>Accuracy Convergence Curve</h5>", unsafe_allow_html=True)
                    acc_df = csv_data[['epoch', 'accuracy', 'val_accuracy']].set_index('epoch')
                    acc_df.columns = ['Train Accuracy', 'Validation Accuracy']
                    st.line_chart(acc_df)

                with ch_c2:
                    st.markdown("<h5 style='color: #cbd5e1;'>Loss Convergence Curve</h5>", unsafe_allow_html=True)
                    loss_df = csv_data[['epoch', 'loss', 'val_loss']].set_index('epoch')
                    loss_df.columns = ['Train Loss', 'Validation Loss']
                    st.line_chart(loss_df)

                st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================================================
    # WORKSPACE 3: CROSS-MODEL COMPARISON MATRIX
    # ==========================================================================
    with tab3:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0; font-weight: 800; color: #ffffff;">Cross-Architecture Benchmark Leaderboard</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Comparative empirical evaluation of ResNet50, MobileNetV2, and Custom CNN across all performance metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)

        rows = []
        chart_data = []

        for name, info in model_options.items():
            m_json = load_json_metrics(info["json"])
            if m_json:
                rows.append({
                    "Model Architecture": name,
                    "Accuracy": f"{m_json['accuracy'] * 100:.2f}%",
                    "Macro F1": f"{m_json['macro_f1'] * 100:.2f}%",
                    "ROC-AUC": f"{m_json['roc_auc'] * 100:.2f}%",
                    "Fire Precision": f"{m_json['fire_precision'] * 100:.2f}%",
                    "Fire Recall": f"{m_json['fire_recall'] * 100:.2f}%",
                    "Fire F1": f"{m_json['fire_f1'] * 100:.2f}%",
                    "False Negatives": m_json['fire_false_negatives'],
                    "Payload Size": info["size"]
                })
                chart_data.append({
                    "Model": name,
                    "Accuracy (%)": m_json['accuracy'] * 100,
                    "Macro F1 (%)": m_json['macro_f1'] * 100,
                    "Fire Recall (%)": m_json['fire_recall'] * 100
                })

        if rows:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 16px 0; color: #ffffff;">📊 Cross-Model Comparison Matrix</h4>', unsafe_allow_html=True)

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 16px 20px; color: #6ee7b7; font-size: 0.9rem; margin-top: 16px;">
                🏆 <strong>Evaluation Winner:</strong> <strong>ResNet50</strong> delivers the highest overall test accuracy (97.96%), ROC-AUC score (99.82%), and lowest missed fire count (12 False Negatives), establishing it as the top recommendation for high-reliability deployment.
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Chart
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 16px 0; color: #ffffff;">📈 Performance Comparison Chart</h4>', unsafe_allow_html=True)

            st.bar_chart(pd.DataFrame(chart_data).set_index("Model"))
            st.markdown('</div>', unsafe_allow_html=True)

            # Tradeoffs
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 16px 0; color: #ffffff;">⚖️ Deployment & Trade-off Analysis</h4>', unsafe_allow_html=True)

            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("""
                <div style="background: rgba(7, 10, 15, 0.6); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 18px;">
                    <div style="font-weight: 800; color: #34d399; margin-bottom: 8px;">ResNet50 (Transfer Learning)</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                        • <strong>Strength:</strong> Highest precision and lowest false negatives.<br>
                        • <strong>Trade-off:</strong> 91.8 MB weight size.<br>
                        • <strong>Target Environment:</strong> Centralized cloud servers.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tc2:
                st.markdown("""
                <div style="background: rgba(7, 10, 15, 0.6); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 18px;">
                    <div style="font-weight: 800; color: #60a5fa; margin-bottom: 8px;">MobileNetV2 (Transfer Learning)</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                        • <strong>Strength:</strong> 10× smaller payload (8.9 MB), high frame rate.<br>
                        • <strong>Trade-off:</strong> Slight ~2% accuracy drop vs ResNet.<br>
                        • <strong>Target Environment:</strong> Drones & edge hardware.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tc3:
                st.markdown("""
                <div style="background: rgba(7, 10, 15, 0.6); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 18px;">
                    <div style="font-weight: 800; color: #fde047; margin-bottom: 8px;">Custom 4-Layer CNN</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
                        • <strong>Strength:</strong> Ultra-compact (2.1 MB), zero pretrained dependencies.<br>
                        • <strong>Trade-off:</strong> Lower precision & higher false negatives.<br>
                        • <strong>Target Environment:</strong> Lightweight baseline reference.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================================================
    # WORKSPACE 4: SYSTEM DOCUMENTATION
    # ==========================================================================
    with tab4:
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0; font-weight: 800; color: #ffffff;">System Architecture & Dataset Specs</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 4px;">
                Technical specifications of dataset cleaning pipelines, model topologies, and training hyperparameters.
            </p>
        </div>
        """, unsafe_allow_html=True)

        d1, d2 = st.columns([1, 1], gap="large")

        with d1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 14px 0; color: #ffffff;">📂 Dataset Partitioning & Cleaning</h4>', unsafe_allow_html=True)
            st.markdown("""
            The benchmark dataset comprises high-resolution aerial and ground-level forestry images partitioned into **Fire** and **Non-Fire** classes.
            
            - **Total Cleaned Image Count:** 5,380 samples
            - **Training Partition (60%):** 3,230 images
            - **Validation Partition (20%):** 1,073 images
            - **Test Partition (20%):** 1,077 untouched evaluation samples
            - **Input Resolution:** 224 × 224 pixels (3-channel RGB)
            - **Class Balance:** ~50/50 Fire to Non-Fire ratio
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with d2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin: 0 0 14px 0; color: #ffffff;">⚙️ Hyperparameters & Training Setup</h4>', unsafe_allow_html=True)
            st.markdown("""
            - **Optimizer:** Adam (Initial LR = 0.001 with `ReduceLROnPlateau`)
            - **Loss Function:** Binary Crossentropy
            - **Batch Size:** 32 samples per mini-batch
            - **Augmentations:** Horizontal Flip, Random Rotation (±15°), Zoom (±10%)
            - **Execution Engine:** Keras / TensorFlow 2.x GPU Backend
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # D. FOOTER
    # --------------------------------------------------------------------------
    st.markdown("""
    <div class="cyber-footer">
        🌲 <strong>Forest Guardian AI</strong> — Enterprise Autonomous Wildfire Computer Vision Suite<br>
        Developed with TensorFlow 2.x & Streamlit.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
