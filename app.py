import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import os

from inference import InferenceEngine, CLASSES
from xai_utils import make_gradcam_heatmap, overlay_heatmap

# Page Configuration
st.set_page_config(
    page_title="X-Skin Diagnostic AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional aesthetic and mobile responsiveness
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .stat-box { padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 20px; }
    .dark-stat-box { padding: 20px; border-radius: 10px; background-color: #262730; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Initialization
@st.cache_resource
def load_engine():
    engine = InferenceEngine(models_dir="models")
    avail, has_rf = engine.get_available_models()
    return engine, avail, has_rf

engine, available_models, has_rf = load_engine()

# OOD Rejection Thresholds
CONFIDENCE_THRESHOLD = 0.50  # Minimum top-class probability
ENTROPY_THRESHOLD = 1.6      # Maximum allowed entropy (uniform over 7 classes = 1.946)

def is_valid_prediction(probs):
    """Check if the prediction looks like a genuine skin lesion or random noise."""
    max_conf = np.max(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return max_conf >= CONFIDENCE_THRESHOLD and entropy <= ENTROPY_THRESHOLD, max_conf, entropy

# Sidebar: Navigation & Author Info
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home (Overview)", "Diagnostic Tool"])
    
    st.markdown("---")
    
    if page == "Diagnostic Tool":
        st.markdown("### Configuration")
        model_options = []
        if available_models:
            model_options.extend(available_models)
            if len(available_models) > 1:
                model_options.append("Ensemble: Weighted Soft-Voting")
                model_options.append("Ensemble: Rank-Based Voting")
                if has_rf:
                    model_options.append("Ensemble: Stacking (Random Forest)")
                    
        if not model_options:
            st.error("No models found in the /models directory.")
            st.stop()
            
        selected_mode = st.selectbox("Select Inference Model", model_options)
        st.markdown("---")
    
    st.markdown("### Developer Information")
    st.markdown("**Anthony Opoku-Acheampong**")
    st.markdown("Student ID: 4245230003")
    st.markdown("BSc Data Science and Analytics")
    st.markdown("Final Year Project")

# ==========================================
# PAGE 1: HOME (LANDING PAGE)
# ==========================================
if page == "Home (Overview)":
    st.title("X-Skin: Advanced Dermatoscopic AI")
    st.markdown("Welcome to the X-Skin Diagnostic AI platform. This application leverages advanced Deep Learning and Ensemble techniques to classify dermatoscopic images into one of seven disease categories.")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### The Problem: Diagnostic Bottlenecks in West Africa")
        st.markdown("""
        While skin cancer incidence in West Africa is lower than in Western nations, it remains a critical public health challenge. In Ghana, conditions like Squamous Cell Carcinoma and Malignant Melanoma are predominant.
        * **Etiological Factors**: Unlike Western populations driven primarily by UV exposure, skin malignancies in West Africa are heavily associated with chronic ulcers, burns, albinism, and the use of hydroquinone-based skin bleaching agents.
        * **Late Presentation**: Due to a severe lack of specialized dermatologists, rural populations often present with late-stage malignancies, drastically reducing survival rates.
        * **Reference**: [World Health Organization (WHO) & Global Cancer Observatory (GCO) Ghana Fact Sheets](https://gco.iarc.who.int/today)
        """)
        st.info("The bottleneck is dermatological expertise. Automated AI diagnosis acts as a critical triage tool, enabling early detection and referral in regions where specialists are unavailable.")
        
    with col2:
        st.markdown("### The Solution: Deep Learning")
        st.markdown("""
        This Final Year Project addresses diagnostic bottlenecks by training multiple Convolutional Neural Networks (CNNs) on the **HAM10000 dataset**, a collection of 10,015 dermatoscopic images. 
        
        Rather than relying on a single model, this system employs **Mathematical Ensembling** (Stacking, Weighted Voting, and Rank-Based Voting) to dramatically reduce false-negative rates and improve diagnostic reliability across all disease classes.
        """)

    st.markdown("---")
    st.markdown("### The 7 Diagnostic Classes")
    st.markdown("The AI is trained to distinguish between the following seven categories. Note that differentiating early Melanoma (MEL) from an Atypical Nevus (NV) is highly complex even for trained dermatologists.")
    
    # Class descriptions
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Melanoma (MEL)**")
        st.markdown("A highly malignant skin cancer derived from melanocytes. Early detection is life-saving.")
        
        st.markdown("**2. Melanocytic Nevus (NV)**")
        st.markdown("Common, benign moles. This is the most common class in the real world and in our dataset.")
        
        st.markdown("**3. Basal Cell Carcinoma (BCC)**")
        st.markdown("The most common form of skin cancer. It grows slowly and rarely spreads, but can be locally destructive.")
        
        st.markdown("**4. Actinic Keratosis (AKIEC)**")
        st.markdown("Pre-cancerous skin lesions caused by sun damage. Can evolve into Squamous Cell Carcinoma.")
        
    with c2:
        st.markdown("**5. Benign Keratosis (BKL)**")
        st.markdown("A broad class that includes seborrheic keratoses and solar lentigines. Completely benign.")
        
        st.markdown("**6. Dermatofibroma (DF)**")
        st.markdown("A benign skin growth that often appears after minor trauma like a bug bite. Very rare in our dataset.")
        
        st.markdown("**7. Vascular Lesions (VASC)**")
        st.markdown("Benign anomalies of the blood vessels, such as cherry angiomas. Usually dark red or purple.")

# ==========================================
# PAGE 2: DIAGNOSTIC TOOL
# ==========================================
elif page == "Diagnostic Tool":
    st.title("Dermatoscopic Image Analysis")
    st.markdown("Upload a dermatoscopic image or use your device camera to receive an AI-assisted diagnostic evaluation.")

    input_method = st.radio("Choose Input Method", ["Upload Image", "Use Camera"], horizontal=True)

    image_file = None
    if input_method == "Upload Image":
        image_file = st.file_uploader("Upload Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    else:
        image_file = st.camera_input("Take a picture")

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Input Image", width=300)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image features..."):
                try:
                    # 1. Inference Logic
                    if selected_mode in available_models:
                        probs, raw_array = engine.predict_single_model(image, selected_mode)
                        xai_model = engine._get_or_load_model(selected_mode)
                    elif selected_mode == "Ensemble: Weighted Soft-Voting":
                        probs = engine.predict_weighted_voting(image)
                        xai_model = engine._get_or_load_model(available_models[0])
                    elif selected_mode == "Ensemble: Rank-Based Voting":
                        probs = engine.predict_rank_based(image)
                        xai_model = engine._get_or_load_model(available_models[0])
                    elif selected_mode == "Ensemble: Stacking (Random Forest)":
                        probs = engine.predict_stacking(image)
                        xai_model = engine._get_or_load_model(available_models[0])
                    
                    # 2. OOD Rejection Gate
                    is_valid, max_conf, entropy = is_valid_prediction(probs)
                    
                    if not is_valid:
                        st.error("**Image Rejected: Not a Valid Dermatoscopic Skin Lesion**")
                        st.markdown(f"""
                        The uploaded image does not appear to contain a recognizable skin lesion. 
                        The model's prediction confidence ({max_conf*100:.1f}%) is too low and/or the probability 
                        distribution is too uniform (entropy: {entropy:.2f}) to produce a reliable diagnosis.
                        
                        **Please ensure:**
                        * The image is a close-up dermatoscopic photograph of a skin lesion.
                        * The image is well-lit and in focus.
                        * The lesion is clearly visible and centered in the frame.
                        """)
                    else:
                        # 3. XAI Logic (Grad-CAM)
                        processed_img, raw_array = engine.preprocess_image(image, available_models[0])
                        heatmap = make_gradcam_heatmap(processed_img, xai_model)
                        overlay = overlay_heatmap(raw_array, heatmap)
                        
                        # Results UI (Tabs)
                        tab_diag, tab_prob, tab_xai, tab_impact = st.tabs(["Diagnosis", "Probabilities", "Explainability (Grad-CAM)", "Project Impact"])
                        
                        with tab_diag:
                            pred_idx = np.argmax(probs)
                            pred_class = CLASSES[pred_idx].upper()
                            confidence = probs[pred_idx] * 100
                            
                            st.markdown(f"### Primary Diagnosis: **{pred_class}**")
                            st.progress(int(confidence))
                            st.markdown(f"**Confidence**: {confidence:.2f}%")
                            
                            if confidence < 75.0:
                                st.warning("Low confidence prediction. Clinical correlation is strongly advised.")
                                
                        with tab_prob:
                            df_probs = pd.DataFrame({'Class': [c.upper() for c in CLASSES], 'Probability': probs * 100})
                            fig = px.bar(df_probs, x='Class', y='Probability', 
                                         title="Class Probability Distribution",
                                         labels={'Probability': 'Confidence (%)'},
                                         color='Probability', color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with tab_xai:
                            st.markdown("### Gradient-weighted Class Activation Mapping (Grad-CAM)")
                            st.markdown("The heatmap highlights the specific pixel regions the Convolutional Neural Network focused on to make its prediction. Red indicates high activation.")
                            st.image(overlay, caption=f"Grad-CAM overlay", width=400)
                            
                        with tab_impact:
                            st.markdown("### Project Context and Clinical Impact")
                            st.markdown("""
                            **The Problem Statement**
                            Skin cancer is highly treatable if detected early, but dermatological expertise is not universally accessible. Furthermore, visual diagnosis of lesions like Melanoma vs. Atypical Nevi is highly complex, even for trained specialists.
                            
                            **The X-Skin Solution**
                            This project utilizes advanced Deep Learning architectures to provide an automated, highly accurate "second opinion". By deploying advanced Ensemble Methods, the system drastically reduces the false-negative rate associated with single-model architectures.
                            
                            **Pros of this Architecture**:
                            * **High Accuracy**: Ensemble methods (like the Random Forest Stacking technique) resolve inter-class ambiguities by learning from the collective strengths of four distinct neural networks.
                            * **Explainability**: The integration of Grad-CAM ensures the model is not a "black box". Doctors can visually verify that the AI is analyzing pathological features (like borders and color asymmetry) rather than background noise.
                            * **Mobile Deployment**: The architecture is designed with mobile endpoints in mind, utilizing lightweight models (MobileNetV2) for fast triage.
                            
                            **Cons & Limitations**:
                            * **Compute Heavy**: The full Stacking Ensemble requires significant RAM to load four CNNs into memory.
                            * **Clinical Data Vacuum**: The model currently operates purely on visual data, omitting critical patient metadata (Age, Sex, Anatomical Location) which would further improve diagnostic accuracy.
                            """)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
