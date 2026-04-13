import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense — Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Mulish:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Mulish', sans-serif; }

.stApp {
    background: #f0f4f0;
    min-height: 100vh;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 10%, rgba(34,139,80,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 90% 90%, rgba(22,101,52,0.10) 0%, transparent 60%);
    z-index: 0;
}
.block-container {
    position: relative;
    z-index: 1;
    max-width: 800px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── hero ── */
.hero {
    background: linear-gradient(135deg, #14532d 0%, #166534 40%, #15803d 100%);
    border-radius: 28px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(20,83,45,0.30);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: #bbf7d0;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.7rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}
.hero-title span { color: #86efac; }
.hero-sub {
    color: rgba(187,247,208,0.75);
    font-size: 0.92rem;
    font-weight: 300;
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── step indicator ── */
.step-row {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
    justify-content: center;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
}
.step-dot {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #d1fae5;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #16a34a;
    box-shadow: 0 2px 8px rgba(22,163,74,0.15);
}
.step-label {
    font-size: 0.6rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #6b7280;
    font-weight: 600;
}
.step-line {
    height: 2px;
    background: linear-gradient(90deg, #86efac, #d1fae5);
    width: 55px;
    margin-bottom: 16px;
}

/* ── card ── */
.card {
    background: #ffffff;
    border: 1px solid #e8f5e9;
    border-radius: 20px;
    padding: 1.8rem 2rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(20,83,45,0.06);
}
.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #16a34a;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, #bbf7d0, transparent);
}

/* widgets */
label[data-testid="stWidgetLabel"] p {
    color: #374151 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div {
    background: #f8fdf9 !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    color: #1f2937 !important;
}
div[data-testid="stNumberInput"] input {
    background: #f8fdf9 !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 10px !important;
    color: #1f2937 !important;
}

/* ── predict button ── */
div[data-testid="stButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #14532d, #16a34a);
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2rem;
    cursor: pointer;
    box-shadow: 0 6px 24px rgba(22,163,74,0.35);
    transition: all 0.3s ease;
    margin-top: 0.8rem;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 36px rgba(22,163,74,0.45);
}

/* ── result ── */
.result-wrap { margin-top: 2rem; animation: riseUp 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
@keyframes riseUp {
    from { opacity:0; transform: translateY(30px) scale(0.95); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}
.result-approved {
    background: linear-gradient(135deg, #14532d, #166534);
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(20,83,45,0.30);
}
.result-rejected {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border-radius: 24px;
    padding: 2.4rem 2rem;
    text-align: center;
    box-shadow: 0 20px 60px rgba(127,29,29,0.30);
}
.result-icon { font-size: 3.2rem; margin-bottom: 0.6rem; }
.result-status {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.result-msg {
    color: rgba(255,255,255,0.65);
    font-size: 0.88rem;
    font-weight: 300;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
}
.conf-wrap { margin-top: 1.6rem; text-align: left; }
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.conf-track {
    background: rgba(255,255,255,0.12);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.conf-fill-green {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #86efac, #22c55e);
    box-shadow: 0 0 12px rgba(134,239,172,0.5);
}
.conf-fill-red {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #fca5a5, #ef4444);
    box-shadow: 0 0 12px rgba(252,165,165,0.4);
}
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1.4rem;
}
.metric-tile {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 0.9rem 0.7rem;
    text-align: center;
}
.mt-val { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #fff; }
.mt-lbl { font-size: 0.62rem; color: rgba(255,255,255,0.45); letter-spacing: 1.5px; text-transform: uppercase; margin-top: 0.2rem; }

.tips-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.tips-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #16a34a;
    margin-bottom: 0.6rem;
}
.tips-text { font-size: 0.83rem; color: #374151; line-height: 1.65; }

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 0.72rem;
    margin-top: 2.5rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m = joblib.load("Logistic_reg.pkl")
    s = joblib.load("scaler.pkl")
    return m, s

try:
    model, scaler = load_artifacts()
    loaded = True
except Exception as e:
    loaded = False
    load_err = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">AI-Powered Decision Engine</div>
  <div class="hero-title">Will Your Loan Get <span>Approved?</span></div>
  <div class="hero-sub">Fill in your financial profile and get an instant eligibility prediction backed by machine learning.</div>
</div>
""", unsafe_allow_html=True)

if not loaded:
    st.error(f"Could not load model files. Place `Logistic_reg.pkl` and `scaler.pkl` in the same folder.\n\n{load_err}")
    st.stop()

# ── Step bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="step-row">
  <div class="step-item"><div class="step-dot">1</div><div class="step-label">Personal</div></div>
  <div class="step-line"></div>
  <div class="step-item"><div class="step-dot">2</div><div class="step-label">Financial</div></div>
  <div class="step-line"></div>
  <div class="step-item"><div class="step-dot">3</div><div class="step-label">Loan</div></div>
  <div class="step-line"></div>
  <div class="step-item"><div class="step-dot">4</div><div class="step-label">Result</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Personal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender    = st.selectbox("Gender", ["Male", "Female"])
    married   = st.selectbox("Marital Status", ["Yes", "No"])
with col2:
    dependents    = st.selectbox("Number of Dependents", [0, 1, 2, 3])
    education     = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
col3, col4 = st.columns(2)
with col3:
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
with col4:
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Financial
# ─────────────────────────────────────────────────────────────────────────────
# st.markdown('<div class="card">', unsafe_allow_html=True)
# st.markdown('<div class="sec-label">Financial Profile</div>', unsafe_allow_html=True)
# col5, col6 = st.columns(2)
# with col5:
#     applicant_income   = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=500)
# with col6:
#     coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, max_value=100000, value=0, step=500)

# # ── fix: black text for radio options ──
# st.markdown("""
# <style>
# div[data-testid="stRadio"] label span p {
#     color: #1f2937 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# credit_history = st.radio(
#     "Credit History",
#     ["Good (1.0) — Meets guidelines", "Poor (0.0) — Does not meet guidelines"]
# )
# credit_val = 1.0 if "Good" in credit_history else 0.0

# st.markdown('</div>', unsafe_allow_html=True)



# SECTION 2 — Financial
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">Financial Profile</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    applicant_income   = st.number_input("Applicant Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=500)
with col6:
    coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, max_value=100000, value=0, step=500)

# ── fix: black text for radio options ──
st.markdown("""
<style>
div[data-testid="stRadio"] label span p {
    color: #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)

credit_history = st.radio(
    "Credit History",
    ["Good (1.0) — Meets guidelines", "Poor (0.0) — Does not meet guidelines"]
)
credit_val = 1.0 if "Good" in credit_history else 0.0

# st.markdown('</div>', unsafe_allow_html=True)

# ── fix: black text for radio options ──
st.markdown("""
<style>
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label span p,
div[data-testid="stRadio"] label div,
div[data-testid="stRadio"] p {
    color: #000000 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Loan Details
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">Loan Details</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)
# with col7:
#     loan_amount = st.number_input("Loan Amount (in thousands $)", min_value=1, max_value=700, value=150, step=5)
with col7:
    loan_amount_rupees = st.number_input("Loan Amount (₹)", min_value=1000, value=150000, step=1000)
    loan_amount = loan_amount_rupees / 1000  # convert to thousands for model
    st.caption(f"Model reads this as: {loan_amount:.1f} thousand")
    
with col8:
    loan_term   = st.selectbox("Loan Term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_features():
    total = applicant_income + coapplicant_income
    vec = {
        'Dependents':              float(dependents),
        'ApplicantIncome':         float(applicant_income),
        'CoapplicantIncome':       float(coapplicant_income),
        'LoanAmount':              float(loan_amount),
        'Loan_Amount_Term':        float(loan_term),
        'Credit_History':          credit_val,
        'Total_income':            float(total),
        'Gender_Female':           1.0 if gender == "Female"       else 0.0,
        'Gender_Male':             1.0 if gender == "Male"         else 0.0,
        'Married_No':              1.0 if married == "No"          else 0.0,
        'Married_Yes':             1.0 if married == "Yes"         else 0.0,
        'Education_Graduate':      1.0 if education == "Graduate"      else 0.0,
        'Education_Not Graduate':  1.0 if education == "Not Graduate"  else 0.0,
        'Self_Employed_No':        1.0 if self_employed == "No"   else 0.0,
        'Self_Employed_Yes':       1.0 if self_employed == "Yes"  else 0.0,
        'Property_Area_Rural':     1.0 if property_area == "Rural"     else 0.0,
        'Property_Area_Semiurban': 1.0 if property_area == "Semiurban" else 0.0,
        'Property_Area_Urban':     1.0 if property_area == "Urban"     else 0.0,
    }
    return np.array([[vec[f] for f in scaler.feature_names_in_]])

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if st.button("Check Loan Eligibility"):
    with st.spinner("Analysing your profile..."):
        try:
            X     = build_features()
            X_sc  = scaler.transform(X)
            pred  = model.predict(X_sc)[0]
            proba = model.predict_proba(X_sc)[0]
            conf  = round(float(max(proba)) * 100, 1)
            total = applicant_income + coapplicant_income
            emi   = round((loan_amount * 1000) / loan_term, 0) if loan_term else 0
            dti   = round((emi / total * 100), 1) if total > 0 else 0

            if pred == 1:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-approved">
                    <div class="result-icon">✅</div>
                    <div class="result-status">Loan Approved!</div>
                    <div class="result-msg">Great news — based on your profile, you are likely eligible for this loan. A lender will do a final review before disbursement.</div>
                    <div class="conf-wrap">
                      <div class="conf-label"><span>Approval Confidence</span><span>{conf}%</span></div>
                      <div class="conf-track"><div class="conf-fill-green" style="width:{conf}%;"></div></div>
                    </div>
                    <div class="metrics-grid">
                      <div class="metric-tile"><div class="mt-val">${total:,}</div><div class="mt-lbl">Total Income</div></div>
                      <div class="metric-tile"><div class="mt-val">${emi:,.0f}</div><div class="mt-lbl">Est. Monthly EMI</div></div>
                      <div class="metric-tile"><div class="mt-val">{dti}%</div><div class="mt-lbl">Debt-to-Income</div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tips-card">
                  <div class="tips-title">Next Steps</div>
                  <div class="tips-text">Gather your income proof, ID documents, and property papers. A debt-to-income ratio below 40% significantly improves your final approval chances. Maintaining a clean credit history going forward will strengthen your position.</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-rejected">
                    <div class="result-icon">❌</div>
                    <div class="result-status">Not Approved</div>
                    <div class="result-msg">Based on your current profile, approval is unlikely. This is not final — improving a few key factors can change the outcome.</div>
                    <div class="conf-wrap">
                      <div class="conf-label"><span>Rejection Confidence</span><span>{conf}%</span></div>
                      <div class="conf-track"><div class="conf-fill-red" style="width:{conf}%;"></div></div>
                    </div>
                    <div class="metrics-grid">
                      <div class="metric-tile"><div class="mt-val">${total:,}</div><div class="mt-lbl">Total Income</div></div>
                      <div class="metric-tile"><div class="mt-val">${emi:,.0f}</div><div class="mt-lbl">Est. Monthly EMI</div></div>
                      <div class="metric-tile"><div class="mt-val">{dti}%</div><div class="mt-lbl">Debt-to-Income</div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="tips-card">
                  <div class="tips-title">How to Improve Your Chances</div>
                  <div class="tips-text">The most impactful change is improving your credit history — it is the strongest predictor in this model. Reducing the loan amount or extending the term lowers your EMI burden. Adding a co-applicant with steady income can significantly boost eligibility.</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  LoanSense &nbsp;·&nbsp; Logistic Regression Model &nbsp;·&nbsp; Built with Streamlit<br>
  For informational purposes only. Does not constitute financial advice.
</div>
""", unsafe_allow_html=True)