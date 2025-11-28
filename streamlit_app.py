import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False
if 'result' not in st.session_state:
    st.session_state.result = None

# Theme toggle
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# CSS - Dynamic theme
def get_css():
    if st.session_state.theme == 'dark':
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 25px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        }
        
        .main-header h1 {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(15px);
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            margin: 1rem 0;
            color: white;
        }
        
        .input-section {
            background: rgba(255,255,255,0.08);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
        }
        
        .result-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 3rem;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(255,107,107,0.4);
            animation: slideUp 0.8s ease-out;
        }
        
        .result-low {
            background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
            color: white;
            padding: 3rem;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(81,207,102,0.4);
            animation: slideUp 0.8s ease-out;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            border: none;
            width: 100%;
            height: 55px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102,126,234,0.5);
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 25px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 20px 60px rgba(102,126,234,0.3);
        }
        
        .main-header h1 {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0;
        }
        
        .card {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .input-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        
        .result-high {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 3rem;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(255,107,107,0.3);
            animation: slideUp 0.8s ease-out;
        }
        
        .result-low {
            background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
            color: white;
            padding: 3rem;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(81,207,102,0.3);
            animation: slideUp 0.8s ease-out;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-weight: 600;
            font-size: 1.1rem;
            border: none;
            width: 100%;
            height: 55px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102,126,234,0.4);
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """

st.markdown(get_css(), unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, encoder
    except:
        return None, None, None

model, scaler, encoder = load_models()

# Sidebar
with st.sidebar:
    st.markdown("# 🏥 **MediPredict AI**")
    st.markdown("### Professional Health Screening")
    
    if st.button("🌙 Toggle Theme"):
        toggle_theme()
        st.rerun()
    
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🔬 Screening", "ℹ️ About"])
    
    theme_text = "🌙 Dark" if st.session_state.theme == 'dark' else "☀️ Light"
    st.caption(f"Theme: {theme_text}")

# HOME PAGE
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MediPredict AI</h1>
        <p>AI-Powered Professional Health Screening</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 style="color: #667eea;">🎯 89%+</h2>
            <h3>Accuracy</h3>
            <p>Advanced XGBoost algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2 style="color: #667eea;">⚡ Instant</h2>
            <h3>Results</h3>
            <p>Real-time predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h2 style="color: #667eea;">🔒 Secure</h2>
            <h3>Privacy</h3>
            <p>No data storage</p>
        </div>
        """, unsafe_allow_html=True)

# SCREENING PAGE
elif page == "🔬 Screening":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Professional Screening</h1>
        <p>Complete the form and click "Analyze" for AI prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model files missing. Contact support.")
        st.stop()
    
    # Reset form
    if 'screening_page' not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.result = None
        st.session_state.screening_page = True
    
    # Input form
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 👤 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 40)
        
        st.markdown("### 🚬 Lifestyle")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        
        st.markdown("### 🏥 Medical")
        chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
        allergy = st.selectbox("Allergies", ["No", "Yes"])
    
    with col2:
        st.markdown("### 🩺 Symptoms")
        yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        
        st.markdown("### 🫁 Respiratory")
        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SUBMIT BUTTON (Only shows results after clicking)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 ANALYZE RISK", type="primary", use_container_width=True):
            with st.spinner("🔬 AI analyzing your health data..."):
                time.sleep(1.5)
                
                # Prediction
                data = pd.DataFrame([{
                    'GENDER': 1 if gender == 'Male' else 0,
                    'AGE': age,
                    'SMOKING': 1 if smoking == 'Yes' else 0,
                    'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
                    'ANXIETY': 1 if anxiety == 'Yes' else 0,
                    'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
                    'CHRONIC_DISEASE': 1 if chronic_disease == 'Yes' else 0,
                    'FATIGUE': 1 if fatigue == 'Yes' else 0,
                    'ALLERGY': 1 if allergy == 'Yes' else 0,
                    'WHEEZING': 1 if wheezing == 'Yes' else 0,
                    'ALCOHOL_CONSUMING': 1 if alcohol == 'Yes' else 0,
                    'COUGHING': 1 if coughing == 'Yes' else 0,
                    'SHORTNESS_OF_BREATH': 1 if shortness_breath == 'Yes' else 0,
                    'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
                    'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
                }])
                
                # Features (match training exactly)
                data['RESPIRATORY'] = data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + data['WHEEZING'] + data['CHEST_PAIN']
                data['LIFESTYLE'] = data['SMOKING'] + data['ALCOHOL_CONSUMING']
                data['SYMPTOMS'] = (data['YELLOW_FINGERS'] + data['CHRONIC_DISEASE'] + data['FATIGUE'] + 
                                   data['WHEEZING'] + data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + 
                                   data['SWALLOWING_DIFFICULTY'] + data['CHEST_PAIN'])
                
                scaled = scaler.transform(data)
                pred = model.predict(scaled)[0]
                proba = model.predict_proba(scaled)[0]
                
                result = encoder.inverse_transform([pred])[0]
                risk = proba[1] * 100
                confidence = proba[pred] * 100
                
                st.session_state.result = (result, risk, confidence)
                st.session_state.form_submitted = True
            
            st.success("✅ Analysis complete!")
    
    # Results (ONLY after submit)
    if st.session_state.form_submitted and st.session_state.result:
        result, risk, confidence = st.session_state.result
        
        st.markdown("## 📊 AI Analysis Results")
        
        if result == "YES":
            st.markdown(f"""
            <div class="result-high">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <h3>Risk Level: <strong>{risk:.1f}%</strong></h3>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.error("""
            ### 🚨 IMMEDIATE ACTIONS:
            - 🔴 Consult oncologist immediately
            - 🔴 Schedule CT scan/biopsy
            - 🔴 Stop smoking completely
            - 🔴 Prepare medical records
            """)
        else:
            st.markdown(f"""
            <div class="result-low">
                <h2>✅ LOW RISK</h2>
                <h3>Risk Level: <strong>{risk:.1f}%</strong></h3>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("""
            ### ✅ HEALTH RECOMMENDATIONS:
            - ✅ Annual check-ups
            - ✅ Healthy lifestyle
            - ✅ Regular exercise
            - ✅ Avoid risk factors
            """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Level", f"{risk:.1f}%")
        col2.metric("Confidence", f"{confidence:.1f}%")
        col3.metric("Status", "HIGH" if result == "YES" else "LOW")
        
        # Progress bar
        st.markdown("### 🎯 Risk Gauge")
        st.progress(min(risk/100, 1.0))
        
        # Download report
        st.markdown("---")
        report_data = {
            'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'Result': result,
            'Risk_Percent': f"{risk:.1f}%",
            'Confidence': f"{confidence:.1f}%",
            'Age': age,
            'Gender': gender,
            'Smoking': smoking
        }
        df_report = pd.DataFrame([report_data])
        csv = df_report.to_csv(index=False)
        st.download_button(
            "📥 Download Report",
            csv,
            f"medipredict_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.info("👆 **Click "🚀 ANALYZE RISK"** to get your AI health assessment")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About MediPredict AI</h1>
        <p>Professional AI Health Screening Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Mission
    Democratizing access to AI-powered early disease detection.
    
    ## 🤖 Technology
    - **Algorithm**: XGBoost Machine Learning
    - **Accuracy**: 89%+ validated
    - **Features**: 18 clinical parameters
    - **Preprocessing**: SMOTE balancing
    
    ## ⚠️ Disclaimer
    **Educational tool only:**
    - ❌ Not medical diagnosis
    - ❌ Not FDA approved
    - ✅ Screening awareness only
    - 🩺 Consult physicians always
    
    ---
    © 2025 MediPredict AI
    """)

# Footer
st.markdown("---")
st.markdown("*© 2025 MediPredict AI | Educational & Screening Tool Only*")
