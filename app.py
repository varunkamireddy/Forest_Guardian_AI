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

# 1. Page Configuration
st.set_page_config(
    page_title="Forest Guardian AI",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2e7d32;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .fire-alert {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 15px;
        border-radius: 6px;
        color: #b71c1c;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 15px;
        border-radius: 6px;
        color: #1b5e20;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #856404;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Model Caching Helper
@st.cache_resource
def load_cached_model(filepath):
    """Load Keras model with caching to prevent repeated loading."""
    if not os.path.exists(filepath):
        return None
    return load_model(filepath)

def load_json_metrics(filepath):
    """Load pre-calculated test metrics from JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

# Main App Execution
def main():
    # Header
    st.markdown('<div class="main-title">🌲 Forest Guardian AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An AI-Based Forest Wildfire Image Detection System</div>', unsafe_allow_html=True)

    # Academic Disclaimer Callout
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic & Research Disclaimer:</strong> This system is a research-based image classification application designed for academic evaluation. It is <strong>NOT</strong> connected to live satellite feeds or real-time sensor networks, and should not replace professional emergency wildfire monitoring infrastructure.
    </div>
    <br>
    """, unsafe_allow_html=True)

    # 3. Sidebar Model Selector
    st.sidebar.title("⚙️ Model Configuration")
    
    model_options = {
        "Custom CNN": {
            "file": "custom_cnn_model.h5",
            "json": "custom_cnn_test_metrics.json",
            "desc": "Custom convolutional neural network baseline",
            "preprocess": "internal"
        },
        "MobileNetV2": {
            "file": "mobilenet_model.h5",
            "json": "mobilenet_test_metrics.json",
            "desc": "ImageNet-pretrained lightweight transfer-learning model",
            "preprocess": "mobilenet"
        },
        "ResNet50": {
            "file": "resnet_model.h5",
            "json": "resnet_test_metrics.json",
            "desc": "ImageNet-pretrained deeper transfer-learning model",
            "preprocess": "resnet"
        }
    }

    selected_model_name = st.sidebar.selectbox(
        "Select Model Architecture:",
        options=list(model_options.keys()),
        index=2  # Default to ResNet50
    )

    model_info = model_options[selected_model_name]
    
    st.sidebar.markdown(f"**Model Description:**\n*{model_info['desc']}*")
    st.sidebar.markdown(f"**Model File:** `{model_info['file']}`")

    # Load Model
    with st.spinner(f"Loading {selected_model_name}..."):
        try:
            model = load_cached_model(model_info["file"])
        except Exception as e:
            st.error(f"Error loading model '{model_info['file']}': {str(e)}")
            model = None

    if model is None:
        st.error(f"Model file `{model_info['file']}` not found or corrupted. Please ensure model training has completed.")
        return

    # Tabs for App Functionality
    tab1, tab2, tab3 = st.tabs(["🔥 Image Inference", "📊 Model Evaluation", "🏆 Model Comparison"])

    # ==================================================
    # TAB 1: IMAGE INFERENCE
    # ==================================================
    with tab1:
        st.subheader("Upload Image for Wildfire Detection")
        
        uploaded_file = st.file_uploader(
            "Choose a forest or aerial image (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"]
        )

        col1, col2 = st.columns([1, 1])

        if uploaded_file is not None:
            try:
                # Load PIL Image
                img = Image.open(uploaded_file).convert("RGB")

                with col1:
                    st.image(img, caption="Uploaded Image", use_container_width=True)

                # Optional Ground Truth Selection
                ground_truth = st.radio(
                    "Optional Ground Truth Label:",
                    options=["Not specified", "Fire", "Non-Fire"],
                    horizontal=True
                )

                # Preprocessing
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32)

                if model_info["preprocess"] == "internal":
                    # Custom CNN: internal Rescaling(1./255), pass [0, 255] float array
                    input_tensor = np.expand_dims(img_array.copy(), axis=0)
                elif model_info["preprocess"] == "mobilenet":
                    # MobileNetV2 preprocess_input
                    input_tensor = np.expand_dims(mobilenet_preprocess(img_array.copy()), axis=0)
                elif model_info["preprocess"] == "resnet":
                    # ResNet50 preprocess_input
                    input_tensor = np.expand_dims(resnet_preprocess(img_array.copy()), axis=0)

                # Inference
                prediction_prob = float(model.predict(input_tensor, verbose=0)[0][0])

                # Class mapping: 0 = Fire, 1 = Non-Fire
                # Output = P(Non-Fire)
                non_fire_prob = prediction_prob
                fire_prob = 1.0 - prediction_prob

                if prediction_prob < 0.5:
                    predicted_class = "Fire"
                    confidence = fire_prob * 100.0
                else:
                    predicted_class = "Non-Fire"
                    confidence = non_fire_prob * 100.0

                with col2:
                    st.markdown("### Prediction Result")
                    
                    if predicted_class == "Fire":
                        st.markdown(f'<div class="fire-alert">🔥 WILDFIRE DETECTED</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert">🌲 NON-FIRE / SAFE</div>', unsafe_allow_html=True)

                    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
                    st.markdown(f"**Selected Model:** `{selected_model_name}`")

                    st.write("---")
                    st.markdown("**Probability Distribution:**")
                    st.write(f"🔥 Fire Probability: **{fire_prob * 100:.2f}%**")
                    st.progress(float(fire_prob))
                    
                    st.write(f"🌲 Non-Fire Probability: **{non_fire_prob * 100:.2f}%**")
                    st.progress(float(non_fire_prob))

                    # Compare Ground Truth if specified
                    if ground_truth != "Not specified":
                        st.write("---")
                        if predicted_class == ground_truth:
                            st.success(f"✅ Prediction matches Ground Truth ({ground_truth})!")
                        else:
                            st.warning(f"⚠️ Prediction ({predicted_class}) differs from Ground Truth ({ground_truth}).")

            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
        else:
            st.info("Please upload an image above to run classification.")

    # ==================================================
    # TAB 2: MODEL EVALUATION (Test Dataset Metrics)
    # ==================================================
    with tab2:
        st.subheader(f"Test Set Evaluation Metrics — {selected_model_name}")
        st.write("These metrics represent performance on the untouched test dataset.")

        metrics = load_json_metrics(model_info["json"])
        
        if metrics is None:
            st.error(f"Test metrics file `{model_info['json']}` not found.")
        else:
            # Metric Columns
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m2.metric("Macro F1-Score", f"{metrics['macro_f1'] * 100:.2f}%")
            m3.metric("ROC-AUC Score", f"{metrics['roc_auc'] * 100:.2f}%")
            m4.metric("Fire Recall", f"{metrics['fire_recall'] * 100:.2f}%")

            st.write("---")
            m5, m6, m7 = st.columns(3)
            m5.metric("Fire Precision", f"{metrics['fire_precision'] * 100:.2f}%")
            m6.metric("Fire F1-Score", f"{metrics['fire_f1'] * 100:.2f}%")
            m7.metric("Fire False Negatives", f"{metrics['fire_false_negatives']}")

            st.write("---")
            st.markdown("### Confusion Matrix (Test Dataset)")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["True Fire (0)", "True Non-Fire (1)"],
                columns=["Pred Fire (0)", "Pred Non-Fire (1)"]
            )
            st.table(cm_df)

    # ==================================================
    # TAB 3: MODEL COMPARISON
    # ==================================================
    with tab3:
        st.subheader("Cross-Model Comparison Matrix")
        st.write("Comparative evaluation of all three models on the untouched test dataset:")

        comp_data = []
        for name, info in model_options.items():
            m_data = load_json_metrics(info["json"])
            if m_data:
                comp_data.append({
                    "Model": name,
                    "Accuracy": f"{m_data['accuracy'] * 100:.2f}%",
                    "Macro F1": f"{m_data['macro_f1'] * 100:.2f}%",
                    "ROC-AUC": f"{m_data['roc_auc'] * 100:.2f}%",
                    "Fire Precision": f"{m_data['fire_precision'] * 100:.2f}%",
                    "Fire Recall": f"{m_data['fire_recall'] * 100:.2f}%",
                    "Fire F1": f"{m_data['fire_f1'] * 100:.2f}%",
                    "Fire False Negatives": m_data['fire_false_negatives']
                })

        if comp_data:
            comp_df = pd.DataFrame(comp_data)
            st.table(comp_df)

            st.markdown("""
            <div class="safe-alert" style="font-size: 1rem; margin-top: 15px;">
                🏆 <strong>Conclusion:</strong> ResNet50 achieved the highest overall test accuracy and ROC-AUC among the three evaluated models.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Unable to load test metrics for comparison.")

if __name__ == "__main__":
    main()
