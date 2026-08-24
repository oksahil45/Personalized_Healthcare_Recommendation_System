
import os
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MediScope | Healthcare Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Cleaned_Dataset.csv")
MEDICINE_PATH = os.path.join(BASE_DIR, "data", "medicine_database.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.joblib")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #07111F;
    --navy2: #0B233D;
    --blue: #38BDF8;
    --blue-dark: #0369A1;
    --text: #0F172A;
    --muted: #64748B;
    --line: #DCE6F0;
    --surface: #FFFFFF;
    --bg: #F5F8FC;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: var(--bg);
}
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 1px solid #172A43;
}
[data-testid="stSidebar"] * {
    color: #EAF3FF !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    font-weight: 600;
}
.brand {
    padding: 10px 0 22px;
    border-bottom: 1px solid #29405D;
    margin-bottom: 18px;
}
.brand-title {
    font-size: 28px;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -1px;
}
.brand-subtitle {
    font-size: 11px;
    color: #9DB2C9 !important;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}
.hero {
    background: linear-gradient(135deg, #07111F 0%, #0B2947 100%);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
    border: 1px solid #173B5B;
}
.hero-kicker {
    color: #7DD3FC !important;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.hero-title {
    color: #FFFFFF !important;
    font-size: 36px;
    font-weight: 800;
    margin: 7px 0 8px;
    letter-spacing: -1.4px;
}
.hero-text {
    color: #C7D7E8 !important;
    font-size: 14px;
    max-width: 900px;
    line-height: 1.7;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 18px;
    min-height: 112px;
    box-shadow: 0 3px 12px rgba(7,17,31,.04);
}
.metric-label {
    color: #64748B !important;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}
.metric-value {
    color: #07111F !important;
    font-size: 28px;
    font-weight: 800;
    margin-top: 7px;
}
.metric-note {
    color: #64748B !important;
    font-size: 11px;
    margin-top: 3px;
}
.section-title {
    color: #0F172A !important;
    font-size: 25px;
    font-weight: 800;
    margin: 20px 0 5px;
}
.section-subtitle {
    color: #64748B !important;
    font-size: 14px;
    margin-bottom: 18px;
}
.info-box {
    background: #EDF8FF;
    border: 1px solid #BFE7FA;
    border-radius: 12px;
    padding: 15px 17px;
    color: #173B57 !important;
    font-size: 13px;
    line-height: 1.65;
}
.warning-box {
    background: #FFF8E7;
    border: 1px solid #F5D98A;
    border-radius: 12px;
    padding: 15px 17px;
    color: #713F12 !important;
    font-size: 13px;
    line-height: 1.65;
}
.result-card {
    background: #FFFFFF;
    border: 1px solid #DCE6F0;
    border-radius: 16px;
    padding: 22px;
    margin-top: 12px;
    box-shadow: 0 3px 12px rgba(7,17,31,.05);
}
.result-disease {
    font-size: 30px;
    font-weight: 800;
    color: #07111F !important;
    margin: 5px 0;
}
.small-muted {
    color: #64748B !important;
    font-size: 12px;
}
.tag {
    display: inline-block;
    background: #E0F2FE;
    color: #075985 !important;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin: 2px;
}
.footer {
    margin-top: 45px;
    padding: 18px 0;
    border-top: 1px solid #DCE6F0;
    color: #718096 !important;
    font-size: 11px;
    text-align: center;
}
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E3EAF2;
    border-radius: 16px;
    padding: 20px;
}
.stButton > button {
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}
h1, h2, h3, h4, p, li, label {
    color: #0F172A;
}
[data-testid="stDataFrame"] {
    border: 1px solid #DCE6F0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Loaders
# -----------------------------
@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

@st.cache_data
def load_medicine_database():
    if not os.path.exists(MEDICINE_PATH):
        return None
    try:
        with open(MEDICINE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

df = load_dataset()
model = load_model()
medicine_db = load_medicine_database()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-title">MEDISCOPE</div>
        <div class="brand-subtitle">Healthcare Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Patient Assessment", "Recommendations",
         "Analytics", "Model Performance", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Prototype • Data Science / ML")
    st.caption("Reference-based healthcare decision support")

# -----------------------------
# Dataset stats
# -----------------------------
if not df.empty:
    total_records = len(df)
    unique_records = len(df.drop_duplicates())
    duplicate_records = int(df.duplicated().sum())
    disease_col = "disease" if "disease" in df.columns else None
    disease_count = int(df[disease_col].nunique()) if disease_col else 0

    def positive_rate(column):
        if column not in df.columns or len(df) == 0:
            return 0
        values = df[column].astype(str).str.lower().str.strip()
        positive = values.isin(["yes", "true", "1", "positive"]).sum()
        return round((positive / len(df)) * 100, 1)

    fever_rate = positive_rate("fever")
    breathing_rate = positive_rate("difficulty_breathing")
else:
    total_records = unique_records = duplicate_records = disease_count = 0
    fever_rate = breathing_rate = 0
    disease_col = None

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">Personalized Healthcare & Medicine Recommendation System</div>
        <div class="hero-title">Healthcare intelligence, in one workspace.</div>
        <div class="hero-text">
            A data science prototype for patient assessment, disease-category prediction,
            reference-based care information and healthcare analytics. The system is
            intended for educational and portfolio demonstration, not clinical diagnosis
            or prescription.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    cards = [
        ("Patient records", f"{total_records:,}", "Rows in supplied dataset"),
        ("Disease classes", f"{disease_count:,}", "Unique target categories"),
        ("Fever records", f"{fever_rate}%", "Positive symptom rate"),
        ("Breathing difficulty", f"{breathing_rate}%", "Positive symptom rate"),
    ]
    for col, (label, value, note) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Exploration of the supplied healthcare dataset.</div>', unsafe_allow_html=True)

    if not df.empty:
        left, right = st.columns([1.25, 1])
        with left:
            if disease_col:
                top = df[disease_col].value_counts().head(10).reset_index()
                top.columns = ["Disease", "Records"]
                fig = px.bar(top.sort_values("Records"), x="Records", y="Disease",
                             orientation="h", title="Most represented disease categories")
                fig.update_layout(template="plotly_white", height=410,
                                  margin=dict(l=10,r=20,t=55,b=10))
                st.plotly_chart(fig, use_container_width=True)
        with right:
            if "age" in df.columns:
                fig = px.histogram(df, x="age", nbins=20, title="Patient age distribution")
                fig.update_layout(template="plotly_white", height=410,
                                  margin=dict(l=10,r=20,t=55,b=10))
                st.plotly_chart(fig, use_container_width=True)

        q1, q2, q3 = st.columns(3)
        q1.metric("Unique rows", f"{unique_records:,}")
        q2.metric("Exact duplicates", f"{duplicate_records:,}")
        q3.metric("Features", f"{df.shape[1]:,}")

# -----------------------------
# Patient Assessment
# -----------------------------
elif page == "Patient Assessment":
    st.markdown('<div class="section-title">Patient Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter patient attributes and run the trained Random Forest pipeline.</div>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset could not be loaded. Verify data/Cleaned_Dataset.csv exists in the repository.")
        st.stop()
    if model is None:
        st.error("Model could not be loaded. Verify models/disease_model.joblib exists in the repository.")
        st.stop()

    required = ["fever","cough","fatigue","difficulty_breathing","age",
                "gender","blood_pressure","cholesterol_level"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {', '.join(missing)}")
        st.stop()

    st.markdown("""
    <div class="info-box">
        This assessment is a machine-learning demonstration. The result is a model
        prediction, not a medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    with st.form("assessment_form"):
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            gender_options = sorted(df["gender"].dropna().astype(str).unique().tolist())
            gender = st.selectbox("Gender", gender_options)
            fever = st.selectbox("Fever", ["Yes", "No"])
            cough = st.selectbox("Cough", ["Yes", "No"])
        with b:
            fatigue = st.selectbox("Fatigue", ["Yes", "No"])
            breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
            bp_values = sorted(df["blood_pressure"].dropna().astype(str).unique().tolist())
            cholesterol_values = sorted(df["cholesterol_level"].dropna().astype(str).unique().tolist())
            blood_pressure = st.selectbox("Blood Pressure Category", bp_values)
            cholesterol = st.selectbox("Cholesterol Category", cholesterol_values)

        submitted = st.form_submit_button("Run Health Assessment", use_container_width=True, type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "fever": fever, "cough": cough, "fatigue": fatigue,
            "difficulty_breathing": breathing, "age": age, "gender": gender,
            "blood_pressure": blood_pressure, "cholesterol_level": cholesterol,
        }])

        try:
            prediction = str(model.predict(input_df)[0])
            ranking = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_
                ranking = pd.DataFrame({"Disease": classes, "Confidence": probabilities})
                ranking = ranking.sort_values("Confidence", ascending=False).head(5)
                confidence = float(ranking.iloc[0]["Confidence"]) * 100
            else:
                confidence = None
                ranking = pd.DataFrame({"Disease":[prediction], "Confidence":[np.nan]})

            symptom_score = sum(str(v).lower() == "yes" for v in [fever,cough,fatigue,breathing])
            risk = "High" if symptom_score >= 3 else ("Medium" if symptom_score == 2 else "Low")

            st.session_state["last_prediction"] = prediction

            st.markdown(f"""
            <div class="result-card">
                <div class="small-muted">TOP MODEL PREDICTION</div>
                <div class="result-disease">{prediction}</div>
                <div class="small-muted">Model confidence: {confidence:.1f}%</div>
                <br>
                <div class="info-box">
                    <strong>Prototype symptom profile: {risk}</strong><br>
                    This indicator is calculated from the entered symptoms and is not a
                    clinical risk assessment.
                </div>
            </div>
            """, unsafe_allow_html=True)

            chart = ranking.copy()
            chart["Confidence"] = chart["Confidence"] * 100
            fig = px.bar(chart.sort_values("Confidence"), x="Confidence", y="Disease",
                         orientation="h", text="Confidence", title="Top 5 model predictions")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(template="plotly_white", height=380,
                              margin=dict(l=10,r=40,t=55,b=10),
                              xaxis_title="Model confidence (%)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")

# -----------------------------
# Recommendations
# -----------------------------
elif page == "Recommendations":
    st.markdown('<div class="section-title">Medicine & Care References</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Reference information connected to the model output.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <strong>Educational use only.</strong> The medicine database is supplied project
        reference data. It is not a prescription, and users should consult a qualified
        healthcare professional before taking or changing medication.
    </div>
    """, unsafe_allow_html=True)

    prediction = st.session_state.get("last_prediction")
    if prediction:
        st.markdown(f'<div class="info-box">Latest model prediction: <strong>{prediction}</strong></div>', unsafe_allow_html=True)
    else:
        st.info("Run a Patient Assessment first to connect references to a prediction.")

    if medicine_db is None:
        st.error("Medicine reference database could not be loaded. Verify data/medicine_database.pkl exists.")
    elif isinstance(medicine_db, dict):
        if prediction and prediction in medicine_db:
            rec = medicine_db[prediction]
            st.markdown(f"### Reference profile: {prediction}")

            if isinstance(rec, dict):
                sections = [
                    ("Medicines in source data", "medicines"),
                    ("Dosage / usage notes in source data", "dosage_instructions"),
                    ("General advice in source data", "advice"),
                    ("Foods to eat", "foods_to_eat"),
                    ("Foods to avoid", "foods_to_avoid"),
                ]
                for title, key in sections:
                    if key in rec:
                        st.markdown(f"#### {title}")
                        vals = rec[key]
                        if isinstance(vals, (list, tuple)):
                            for item in vals:
                                st.write(f"• {item}")
                        else:
                            st.write(vals)

                c1, c2 = st.columns(2)
                with c1:
                    if rec.get("recovery_time"):
                        st.metric("Reference recovery time", str(rec["recovery_time"]))
                with c2:
                    if rec.get("when_to_see_doctor"):
                        st.markdown("**When to seek professional care**")
                        st.write(rec["when_to_see_doctor"])
            else:
                st.write(rec)
        else:
            st.info("No matching medicine reference exists in the supplied database for this prediction.")
            st.markdown("### Available reference categories")
            st.write(", ".join(sorted(map(str, medicine_db.keys()))))
    else:
        st.write(medicine_db)

# -----------------------------
# Analytics
# -----------------------------
elif page == "Analytics":
    st.markdown('<div class="section-title">Healthcare Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Explore patient patterns, symptoms and dataset quality.</div>', unsafe_allow_html=True)

    if df.empty:
        st.error("Dataset could not be loaded.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Patient Patterns", "Symptoms", "Data Quality"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if "age" in df.columns:
                fig = px.histogram(df, x="age", color="gender" if "gender" in df.columns else None,
                                   nbins=20, title="Age distribution")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "risk_level" in df.columns:
                risk_counts = df["risk_level"].astype(str).value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Records"]
                fig = px.pie(risk_counts, names="Risk Level", values="Records",
                             hole=.45, title="Risk-level distribution")
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        if "gender" in df.columns:
            gender_counts = df["gender"].astype(str).value_counts().reset_index()
            gender_counts.columns = ["Gender", "Records"]
            fig = px.bar(gender_counts, x="Gender", y="Records", title="Records by gender")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        symptom_cols = [c for c in ["fever","cough","fatigue","difficulty_breathing"] if c in df.columns]
        rows = []
        for col in symptom_cols:
            values = df[col].astype(str).str.lower().str.strip()
            positive = int(values.isin(["yes","true","1","positive"]).sum())
            rows.append({"Symptom": col.replace("_"," ").title(),
                         "Positive Records": positive,
                         "Positive Rate (%)": round(positive / len(df) * 100, 1)})
        if rows:
            symptom_df = pd.DataFrame(rows)
            fig = px.bar(symptom_df, x="Symptom", y="Positive Rate (%)",
                         text="Positive Rate (%)", title="Positive symptom rate")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(symptom_df, use_container_width=True, hide_index=True)

    with tab3:
        quality = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
            "Unique Values": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "Data Type": [str(df[c].dtype) for c in df.columns],
        })
        q1,q2,q3 = st.columns(3)
        q1.metric("Rows", f"{len(df):,}")
        q2.metric("Exact duplicates", f"{int(df.duplicated().sum()):,}")
        q3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(quality, use_container_width=True, hide_index=True)

# -----------------------------
# Model Performance
# -----------------------------
elif page == "Model Performance":
    st.markdown('<div class="section-title">Model Performance & Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Configuration, training data quality and limitations of the deployed model.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        The deployed classifier is a Random Forest pipeline with categorical encoding,
        numeric imputation and unknown-category handling. The model was trained after
        removing exact duplicate rows from the supplied dataset.
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Raw records", f"{len(df):,}" if not df.empty else "0")
    c2.metric("Training records", f"{len(df.drop_duplicates()):,}" if not df.empty else "0")
    c3.metric("Disease classes", f"{df['disease'].nunique():,}" if "disease" in df.columns else "0")
    c4.metric("Model trees", "300")

    st.markdown("### Model configuration")
    model_info = pd.DataFrame({
        "Component": ["Algorithm","Trees","Max depth","Preprocessing","Categorical encoding","Missing values","Output"],
        "Configuration": [
            "Random Forest Classifier","300","18","ColumnTransformer pipeline",
            "OneHotEncoder (unknown values ignored)","Median / most-frequent imputation",
            "Disease class + probability ranking"
        ],
    })
    st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.markdown("### Why a conventional accuracy score is not shown")
    st.markdown("""
    <div class="warning-box">
        The supplied dataset contains 116 disease classes across 300 unique rows after
        duplicate removal. Many classes have very few examples. A conventional random
        train/test split would therefore produce unstable or misleading class-level
        metrics. The project intentionally does not invent or overstate an accuracy
        number.
    </div>
    """, unsafe_allow_html=True)

    if "disease" in df.columns:
        counts = df.drop_duplicates()["disease"].value_counts()
        st.markdown("### Class distribution")
        st.write(f"Median records per disease class: **{counts.median():.1f}**")
        st.write(f"Disease classes represented by only one unique row: **{int((counts == 1).sum())}**")

# -----------------------------
# About
# -----------------------------
else:
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Personalized Healthcare & Medicine Recommendation System

    This portfolio project combines structured healthcare data, machine learning,
    reference-based care information and interactive analytics in one Streamlit
    application.

    **Core workflow**

    1. Patient attributes are entered.
    2. The same preprocessing pipeline used during training transforms the inputs.
    3. A Random Forest classifier predicts a disease category.
    4. The application ranks the top model predictions by probability.
    5. If a matching entry exists, reference information from the supplied medicine
       database is displayed.
    6. Dataset-level analytics and quality checks provide additional insights.

    **Technology stack**

    Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit • Joblib

    **Portfolio positioning**

    This is an educational machine-learning prototype. It is not a clinically
    validated medical device and does not provide professional diagnosis or
    prescriptions.
    """)
    st.markdown("---")
    st.markdown("**Repository structure**")
    st.code("""data/
  Cleaned_Dataset.csv
  medicine_database.pkl
models/
  disease_model.joblib
app.py
train_model.py
requirements.txt
README.md""")

st.markdown("""
<div class="footer">
    MEDISCOPE • Personalized Healthcare Intelligence • Portfolio Prototype
</div>
""", unsafe_allow_html=True)
