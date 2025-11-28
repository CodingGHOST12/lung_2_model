import streamlit as st
import pandas as pd
import pickle

# Setup
st.set_page_config(page_title="Health AI", page_icon="🏥")

# Title
st.title("🏥 Health Risk Predictor")
st.write("AI-powered disease screening")

# Load models
@st.cache_resource
def load():
    try:
        m = pickle.load(open('lung_cancer_model.pkl', 'rb'))
        s = pickle.load(open('scaler.pkl', 'rb'))
        e = pickle.load(open('label_encoder.pkl', 'rb'))
        return m, s, e
    except:
        return None, None, None

model, scaler, encoder = load()

# Check if loaded
if model is None:
    st.error("❌ Model files not found!")
    st.stop()
else:
    st.success("✅ AI Model Ready")

st.markdown("---")

# Input form
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    smoking = st.radio("Smoking", ["No", "Yes"])
    alcohol = st.radio("Alcohol", ["No", "Yes"])
    peer_pressure = st.radio("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.radio("Chronic Disease", ["No", "Yes"])
    allergy = st.radio("Allergy", ["No", "Yes"])
    yellow_fingers = st.radio("Yellow Fingers", ["No", "Yes"])

with col2:
    anxiety = st.radio("Anxiety", ["No", "Yes"])
    fatigue = st.radio("Fatigue", ["No", "Yes"])
    wheezing = st.radio("Wheezing", ["No", "Yes"])
    coughing = st.radio("Coughing", ["No", "Yes"])
    shortness = st.radio("Shortness of Breath", ["No", "Yes"])
    swallowing = st.radio("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.radio("Chest Pain", ["No", "Yes"])

st.markdown("---")

# Predict
try:
    # Create input
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
        'SHORTNESS_OF_BREATH': 1 if shortness == 'Yes' else 0,
        'SWALLOWING_DIFFICULTY': 1 if swallowing == 'Yes' else 0,
        'CHEST_PAIN': 1 if chest_pain == 'Yes' else 0
    }])
    
    # Add features (MUST match training)
    data['RESPIRATORY'] = data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + data['WHEEZING'] + data['CHEST_PAIN']
    data['LIFESTYLE'] = data['SMOKING'] + data['ALCOHOL_CONSUMING']
    data['SYMPTOMS'] = (data['YELLOW_FINGERS'] + data['CHRONIC_DISEASE'] + data['FATIGUE'] + 
                       data['WHEEZING'] + data['COUGHING'] + data['SHORTNESS_OF_BREATH'] + 
                       data['SWALLOWING_DIFFICULTY'] + data['CHEST_PAIN'])
    
    # Scale
    scaled = scaler.transform(data)
    
    # Predict
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    
    result = encoder.inverse_transform([pred])[0]
    risk = proba[1] * 100
    
    # Show result
    st.subheader("📊 Result")
    
    if result == "YES":
        st.error(f"⚠️ HIGH RISK: {risk:.1f}%")
        st.warning("**Action:** Consult a doctor immediately")
    else:
        st.success(f"✅ LOW RISK: {risk:.1f}%")
        st.info("**Action:** Maintain healthy lifestyle")
    
    # Metrics
    col1, col2 = st.columns(2)
    col1.metric("Risk Level", f"{risk:.1f}%")
    col2.metric("Status", "High" if result == "YES" else "Low")
    
    # Progress bar
    st.progress(min(risk/100, 1.0))
    
except Exception as e:
    st.error(f"Error: {e}")

st.markdown("---")
st.caption("⚠️ For educational purposes only | Not medical advice")
