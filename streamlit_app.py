import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

st.set_page_config(page_title="MediPredict AI", page_icon=":hospital:", layout="wide")

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.main-header { 
    background: rgba(255,255,255,0.1); 
    padding: 2rem; 
    border-radius: 20px; 
    color: white; 
    text-align: center; 
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}
.card { 
    background: rgba(255,255,255,0.15); 
    padding: 2rem; 
    border-radius: 15px; 
    margin: 1rem 0; 
    color: white;
}
.result-high { 
    background: linear-gradient(135deg, #ff6b6b, #ee5a6f); 
    color: white; 
    padding: 2rem; 
    border-radius: 15px; 
    text-align: center; 
}
.result-low { 
    background: linear-gradient(135deg, #51cf66, #40c057); 
    color: white; 
    padding: 2rem; 
    border-radius: 15px; 
    text-align: center; 
}
.stButton > button { 
    background: white; 
    color: #667eea; 
    border-radius: 10px; 
    font-weight: 600; 
    height: 50px;
    font-size: 16px;
}
.metric-container { background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("MediPredict AI")
page = st.sidebar.radio("Navigation:", ["Home", "Screening", "About"])

if page == "Home":
    st.markdown('<div class="main-header"><h1>MediPredict AI</h1><p>Professional AI Health Screening</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card"><h2>89%+</h2><p>Accuracy</p><p>XGBoost ML Model</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h2>Instant</h2><p>Results</p><p>Real-time Analysis</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h2>Secure</h2><p>Privacy</p><p>No Data Stored</p></div>', unsafe_allow_html=True)

elif page == "Screening":
    if model is None:
        st.error("ERROR: Model files not found. Upload lung_cancer_model.pkl, scaler.pkl, label_encoder.pkl")
        st.stop()
    
    st.markdown('<div class="main-header"><h1>Health Risk Assessment</h1><p>Complete form and click Analyze</p></div>', unsafe_allow_html=True)
    
    # Form inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Personal Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age (years)", 18, 100, 40)
        
        st.markdown("### Lifestyle Factors")
        smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol consumption?", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer pressure to smoke/drink?", ["No", "Yes"])
        
        st.markdown("### Medical History")
        chronic_disease = st.selectbox("Chronic diseases?", ["No", "Yes"])
    
    with col2:
        st.markdown("### Current Symptoms")
        yellow_fingers = st.selectbox("Yellow fingers?", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety?", ["No", "Yes"])
        fatigue = st.selectbox("Chronic fatigue?", ["No", "Yes"])
        
        st.markdown("### Respiratory Symptoms")
        wheezing = st.selectbox("Wheezing?", ["No", "Yes"])
        coughing = st.selectbox("Persistent coughing?", ["No", "Yes"])
        shortness_breath = st.selectbox("Shortness of breath?", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing difficulty?", ["No", "Yes"])
        chest_pain = st.selectbox("Chest pain?", ["No", "Yes"])
    
    st.markdown("---")
    
    # ANALYZE BUTTON
    if st.button("ANALYZE MY RISK", use_container_width=True):
        with st.spinner("AI analyzing your health data..."):
            time.sleep(1.2)
            
            # Create input data
            input_data = pd.DataFrame([{
                'GENDER': 1 if gender == 'Male' else 0,
                'AGE': age,
                'SMOKING': 1 if smoking == 'Yes' else 0,
                'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
                'ANXIETY': 1 if anxiety == 'Yes' else 0,
                'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
                'CHRONIC_DISEASE': 1 if chronic_disease == 'Yes' else 0,
                'FATIGUE': 1 if fatigue == 'Yes' else 0,
                'ALLERGY': 0,
                'WHEEZING': 1 if wheezing == 'Yes' else 0,
                'ALCOHOL_CONSUMING': 1 if alcohol == 'Yes' else 0,
                'COUGHING': 1 if coughing == 'Yes' else 0,
                'SHORTNESS_OF_BREATH': 1 if shortness_breath == 'Yes' else 0,
                'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
                'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
            }])
            
            # Feature engineering (MATCHES TRAINING)
            input_data['RESPIRATORY'] = (input_data['COUGHING'] + input_data['SHORTNESS_OF_BREATH'] + 
                                       input_data['WHEEZING'] + input_data['CHEST_PAIN'])
            input_data['LIFESTYLE'] = input_data['SMOKING'] + input_data['ALCOHOL_CONSUMING']
            input_data['SYMPTOMS'] = (input_data['YELLOW_FINGERS'] + input_data['CHRONIC_DISEASE'] + 
                                    input_data['FATIGUE'] + input_data['WHEEZING'] + 
                                    input_data['COUGHING'] + input_data['SHORTNESS_OF_BREATH'] + 
                                    input_data['SWALLOWING_DIFFICULTY'] + input_data['CHEST_PAIN'])
            
            # Predict
            X_scaled = scaler.transform(input_data)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            result = encoder.inverse_transform([prediction])[0]
            risk_score = probability[1] * 100
            
            # Display Results
            st.markdown("## AI Analysis Results")
            
            if result == "YES":
                st.markdown(f'''
                <div class="result-high">
                    <h2>HIGH RISK DETECTED</h2>
                    <h3>Risk Level: {risk_score:.1f}%</h3>
                </div>
                ''', unsafe_allow_html=True)
                st.error("**URGENT:** Consult doctor immediately for diagnostic tests")
            else:
                st.markdown(f'''
                <div class="result-low">
                    <h2>LOW RISK</h2>
                    <h3>Risk Level: {risk_score:.1f}%</h3>
                </div>
                ''', unsafe_allow_html=True)
                st.success("**Good news:** Continue healthy lifestyle and regular checkups")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.1f}%")
            col2.metric("Status", "HIGH RISK" if result == "YES" else "LOW RISK")
            col3.metric("Age", age)
            
            # Progress bar
            st.progress(min(risk_score/100, 1.0))
            
            # Download report
            report = pd.DataFrame([{
                'Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'Result': result,
                'Risk_Percent': f"{risk_score:.1f}%",
                'Age': age,
                'Gender': gender,
                'Smoking': smoking
            }])
            csv = report.to_csv(index=False)
            st.download_button("Download Report", csv, "health_report.csv", "text/csv")
    
    else:
        st.info('Click "ANALYZE MY RISK" button above to get your personalized health assessment')

elif page == "About":
    st.markdown('<div class="main-header"><h1>About MediPredict AI</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Key Features
    
    - **89%+ Accuracy** - XGBoost machine learning model
    - **Instant Results** - Real-time AI predictions  
    - **Privacy First** - No data storage or sharing
    - **Professional Design** - Medical-grade interface
    
    ## ⚠️ Important Disclaimer
    **This is an EDUCATIONAL tool only:**
    
    ❌ Not medical diagnosis  
    ❌ Not FDA approved
    ✅ Early screening awareness
    🩺 Always consult physicians
    
    ## Technology
    - Python + Streamlit
    - XGBoost ML Algorithm
    - SMOTE data balancing
    - Professional UI/UX
    
    © 2025 MediPredict AI
    """)

st.markdown("---")
st.caption("Educational health screening tool - Consult healthcare professionals for diagnosis")
