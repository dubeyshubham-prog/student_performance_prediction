import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ==============================================================================
# 1. PREMIUM PAGE CONFIGURATION (MUST BE FIRST)
# ==============================================================================
st.set_page_config(
    page_title="Enterprise ML Analytics Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. ADVANCED INTERFACE OVERHAUL (Custom CSS Injection)
# ==============================================================================
st.markdown("""
    <style>
    /* Base Background Grid & Micro-Textures */
    .stApp {
        background-color: #090d16;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(124, 58, 237, 0.12) 0px, transparent 50%);
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Sleek Cyberpunk Header Dashboard Banner */
    .premium-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 5px solid #6366f1;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    .premium-header h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #a5b4fc, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }

    /* Form Wrapper Containers */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        box-shadow: 0 30px 60px -10px rgba(0, 0, 0, 0.4);
    }

    /* Interactive Card Dividers */
    .section-banner {
        background: rgba(99, 102, 241, 0.06);
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        margin-top: 1rem;
    }
    .section-banner h4 {
        margin: 0 !important;
        color: #e0e7ff !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }

    /* Custom High-Contrast Input Fields Overrides */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: #1e293b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Dynamic Action Button Styling */
    button[kind="formSubmit"] {
        background: linear-gradient(90deg, #4f46e5 0%, #6d28d9 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    button[kind="formSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4) !important;
    }

    /* Elegant Output Target Node */
    .prediction-card {
        background: radial-gradient(circle at top left, rgba(16, 185, 129, 0.1), transparent);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-metrics {
        font-size: 3.5rem;
        font-weight: 900;
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        margin: 0.5rem 0;
    }

    /* Typography Parameter Enhancements */
    label p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. HELPER FUNCTION TO LOAD PIPELINES CACHED
# ==============================================================================
@st.cache_resource
def load_prediction_artifacts():
    """
    Safely resolves files matching internal workspace tree layout: src/components/artifacts/
    """
    model_path = os.path.join('src', 'components', 'artifacts', 'model.pkl')
    preprocessor_path = os.path.join('src', 'components', 'artifacts', 'preprocessor.pkl')

    model = None
    preprocessor = None

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

    return model, preprocessor


# Execute artifact compilation load
model, preprocessor = load_prediction_artifacts()

# ==============================================================================
# 4. PREMIUM HEADER BANNER DISPLAY
# ==============================================================================
st.markdown("""
    <div class="premium-header">
        <h1>Real-Time Student Performance Inference Portal</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 0;">
            Enterprise Pipeline Engine powered by Automated Grid Optimization Stack
        </p>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. HIGH-CONTRAST SIDEBAR MATRIX STATUS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🗺️ Infrastructure Node")

    if model is not None:
        st.markdown(
            f"""<div style='background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 1rem; border-radius: 8px;'>
                <span style='color: #10b981; font-weight: bold;'>● PIPELINE ACTIVE</span><br>
                <small style='color: #a7f3d0;'>Engine Class: {type(model).__name__}</small>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style='background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; padding: 1rem; border-radius: 8px;'>
                <span style='color: #ef4444; font-weight: bold;'>📴 ARTIFACT DISCONNECTED</span><br>
                <small style='color: #fca5a5;'>Path missing from src/components/artifacts/</small>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 📋 Core Context")
    st.info(
        "Input runtime metrics in the core profile workspace panel. Click execute to build transformation graphs instantly.")

# ==============================================================================
# 6. APPLICATION SUBMISSION CONTROLLER & CORRECTED DATA MAPPING
# ==============================================================================
if model is None:
    st.error(
        "🚨 Critical Link Fault: The model workspace cannot bind 'model.pkl'. Verify pipeline artifacts compilation steps.")
else:
    st.markdown("### 🔧 Student Demographics & Performance Metrics")

    with st.form("prediction_form", clear_on_submit=False):

        # Section 1: Qualitative / Categorical Demographics
        st.markdown('<div class="section-banner"><h4>🏷️ Student Background & Demographics</h4></div>',
                    unsafe_allow_html=True)
        cat_col1, cat_col2, cat_col3 = st.columns(3)
        with cat_col1:
            gender = st.selectbox("Gender", ["female", "male"])
            lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        with cat_col2:
            race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
            test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
        with cat_col3:
            parental_level_of_education = st.selectbox(
                "Parental Level of Education",
                ["bachelor's degree", "some college", "master's degree", "associate's degree", "high school",
                 "some high school"]
            )

        # Section 2: Quantitative / Academic Scores
        st.markdown('<div class="section-banner"><h4>📊 Academic Performance Scores</h4></div>', unsafe_allow_html=True)
        num_col1, num_col2 = st.columns(2)
        with num_col1:
            reading_score = st.number_input("Reading Score (out of 100)", min_value=0.0, max_value=100.0, value=70.0)
        with num_col2:
            writing_score = st.number_input("Writing Score (out of 100)", min_value=0.0, max_value=100.0, value=70.0)

        st.markdown("<br>", unsafe_allow_html=True)

        # Primary Pipeline Compute Executor
        submit_btn = st.form_submit_button("⚡ Run Model Graph Execution", use_container_width=True)

    # ==============================================================================
    # 7. INFERENCE LOGIC GRAPH RUNTIME EXECUTION
    # ==============================================================================
    if submit_btn:
        with st.spinner("Compiling structural dataframe matrices & calling pipeline dependencies..."):
            try:
                # ALL KEYS SYNCHRONIZED EXACTLY WITH THE ML PIPELINE REQUIREMENTS:
                raw_input_data = {
                    "gender": [gender],
                    "race_ethnicity": [race_ethnicity],
                    "parental_level_of_education": [parental_level_of_education],
                    "lunch": [lunch],
                    "test_preparation_course": [test_preparation_course],
                    "reading_score": [reading_score],
                    "writing_score": [writing_score]
                }

                # Convert cleanly into structured DataFrame
                input_df = pd.DataFrame(raw_input_data)

                # Execute Preprocessor Matrix Transforms if Available
                if preprocessor is not None:
                    final_features = preprocessor.transform(input_df)
                else:
                    final_features = input_df.values

                    # Predict Outcome value using Loaded Champion Model
                prediction_result = model.predict(final_features)
                output_val = round(float(prediction_result[0]), 2)

                # Ultra-Clean Interactive High-Fidelity Results Module
                st.markdown(f"""
                    <div class="prediction-card">
                        <h4 style="margin:0; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; font-size: 0.9rem;">
                            Predicted Target Score Result
                        </h4>
                        <div class="prediction-metrics">{output_val}</div>
                        <p style="margin:0; color:#64748b; font-size:0.85rem;">
                            Graph Computation Completed Instantly via Active Memory Matrix
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.balloons()

            except Exception as e:
                st.error(f"Inference Graph Fault Execution Error: {str(e)}")