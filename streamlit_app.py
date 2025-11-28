import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide")

# ============================
# Load model and tools
# ============================
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("lung_cancer_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        encoder = pickle.load(open("label_encoder.pkl", "rb"))
        return model, scaler, encoder
    except:
        return None, None, None

model, scaler, encoder = load_models()

# ============================
# Custom UI Styling
# ============================
st.markdown("""
<style>
body { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #5563DE 0%, #7939A8 100%); }

.header {
    padding: 2rem;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
}
.card {
    background: rgba(255,255,255,0.15);
    padding: 1.6rem;
    border-radius: 16px;
    color: white;
}
.result-high {
    background: #e85959;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
}
.result-low {
    background: #51cf66;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ============================
# Sidebar Navigation
# ============================
st.sidebar.title("MediPredict AI")
page = st.sidebar.radio("Navigate", ["Home", "Health Screening", "About"])

# ============================
# HOME PAGE
# ============================
if page == "Home":
    st.markdown('<div class="header"><h1>MediPredict AI</h1><p>Smart Health Screening</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><h2>89%+</h2><p>Model Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h2>Instant</h2><p>AI Predictions</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card"><h2>Private</h2><p>No Data Storage</p></div>', unsafe_allow_html=True)

# ============================
# SCREENING PAGE
# ============================
elif page == "Health Screening":

    if model is None:
        st.error("Model files missing. Upload lung_cancer_model.pkl, scaler.pkl, label_encoder.pkl")
        st.stop()

    st.markdown('<div class="header"><h1>Health Risk Assessment</h1><p>Provide details to begin analysis</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ---- Left column ----
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 40)

        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Use", ["No", "Yes"])
        peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
        chronic = st.selectbox("Chronic Disease", ["No", "Yes"])

    # ---- Right column ----
    with col2:
        yellow = st.selectbox("Yellow Fingers", ["No", "Yes"])
        anxiety = st.selectbox("Anxiety", ["No", "Yes"])
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])

        wheezing = st.selectbox("Wheezing", ["No", "Yes"])
        coughing = st.selectbox("Coughing", ["No", "Yes"])
        shortness = st.selectbox("Shortness of Breath", ["No", "Yes"])
        swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    st.markdown("---")

    # ===== Predict Button =====
    if st.button("Analyze My Risk", use_container_width=True):

        with st.spinner("Analyzing your information..."):
            time.sleep(1)

            # Input data
            data = pd.DataFrame([{
                "GENDER": 1 if gender == "Male" else 0,
                "AGE": age,
                "SMOKING": 1 if smoking == "Yes" else 0,
                "YELLOW_FINGERS": 1 if yellow == "Yes" else 0,
                "ANXIETY": 1 if anxiety == "Yes" else 0,
                "PEER_PRESSURE": 1 if peer_pressure == "Yes" else 0,
                "CHRONIC_DISEASE": 1 if chronic == "Yes" else 0,
                "FATIGUE": 1 if fatigue == "Yes" else 0,
                "ALLERGY": 0,
                "WHEEZING": 1 if wheezing == "Yes" else 0,
                "ALCOHOL_CONSUMING": 1 if alcohol == "Yes" else 0,
                "COUGHING": 1 if coughing == "Yes" else 0,
                "SHORTNESS_OF_BREATH": 1 if shortness == "Yes" else 0,
                "SWALLOWING_DIFFICULTY": 1 if swallowing == "Yes" else 0,
                "CHEST_PAIN": 1 if chest_pain == "Yes" else 0,
            }])

            # Feature engineering
            data["RESPIRATORY"] = (
                data["COUGHING"] +
                data["SHORTNESS_OF_BREATH"] +
                data["WHEEZING"] +
                data["CHEST_PAIN"]
            )
            data["LIFESTYLE"] = data["SMOKING"] + data["ALCOHOL_CONSUMING"]
            data["SYMPTOMS"] = (
                data["YELLOW_FINGERS"] + data["CHRONIC_DISEASE"] + data["FATIGUE"] +
                data["WHEEZING"] + data["COUGHING"] + data["SHORTNESS_OF_BREATH"] +
                data["SWALLOWING_DIFFICULTY"] + data["CHEST_PAIN"]
            )

            X_scaled = scaler.transform(data)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1] * 100
            label = encoder.inverse_transform([pred])[0]

            st.markdown("### Result Summary")

            if label == "YES":
                st.markdown(f"""
                <div class='result-high'>
                <h2>High Risk Detected</h2>
                <h3>Risk Level: {prob:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                <h2>Low Risk</h2>
                <h3>Risk Level: {prob:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            st.progress(prob / 100)

# ============================
# ABOUT PAGE
# ============================
elif page == "About":
    st.markdown('<div class="header"><h1>About MediPredict AI</h1></div>', unsafe_allow_html=True)
    st.write("""
### What this tool does
- Uses machine learning to evaluate health patterns  
- Gives instant screening feedback  
- Works fully offline and doesn’t store any data  

### Important Notes
This is an educational tool and not a medical diagnostic system.  
Always consult healthcare professionals for real diagnosis.
""")

st.markdown("---")
st.caption("© 2025 MediPredict AI | Educational Use Only")

