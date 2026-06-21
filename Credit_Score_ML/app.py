import os
import pickle
import numpy as np
import streamlit as st

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "meta.pkl")

st.set_page_config(page_title="Credit Score Checker", page_icon="💳", layout="centered")

st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 1.5rem;
    }
    .header h1 { color: #e0f2fe; font-size: 2.2rem; margin: 0; }
    .header p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 1rem; }

    .sec-title {
        font-size: 1.05rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 10px; margin: 1.6rem 0 0.8rem;
    }

    /* Score meter */
    .meter-wrap { text-align: center; margin: 1.5rem 0 0.5rem; }
    .meter-score {
        font-size: 4rem; font-weight: 800; line-height: 1;
    }
    .meter-label { font-size: 1.2rem; font-weight: 600; margin-top: 0.3rem; }

    /* Result cards */
    .card-good {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        border: 2px solid #10b981; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .card-good h2 { color: #064e3b; margin: 0; font-size: 1.7rem; }
    .card-good p  { color: #065f46; margin: 0.4rem 0 0; font-size: 0.95rem; }

    .card-standard {
        background: linear-gradient(135deg, #fef9c3, #fde68a);
        border: 2px solid #f59e0b; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .card-standard h2 { color: #78350f; margin: 0; font-size: 1.7rem; }
    .card-standard p  { color: #92400e; margin: 0.4rem 0 0; font-size: 0.95rem; }

    .card-poor {
        background: linear-gradient(135deg, #fee2e2, #fca5a5);
        border: 2px solid #ef4444; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin-top: 1rem;
    }
    .card-poor h2 { color: #7f1d1d; margin: 0; font-size: 1.7rem; }
    .card-poor p  { color: #991b1b; margin: 0.4rem 0 0; font-size: 0.95rem; }

    .tip-box {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 10px; padding: 1rem 1.2rem;
        margin-top: 1rem; font-size: 0.9rem; color: #0c4a6e; line-height: 1.7;
    }
    .tip-box b { color: #0369a1; display: block; margin-bottom: 0.3rem; }

    .gauge-bar-wrap {
        background: #e2e8f0; border-radius: 999px;
        height: 18px; margin: 0.5rem 0;
        overflow: hidden;
    }
    .gauge-bar {
        height: 100%; border-radius: 999px;
        transition: width 0.5s;
    }
    .footer {
        text-align: center; color: #94a3b8;
        font-size: 0.8rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model 
@st.cache_resource
def load_all():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    with open(META_PATH,   "rb") as f: meta   = pickle.load(f)
    return model, scaler, meta

model, scaler, meta = load_all()

st.markdown("""
<div class="header">
    <h1>💳 Credit Score Checker</h1>
    <p>AI-powered credit score prediction — Poor / Standard / Good</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(f"⚠️ Model not found! Run in VS Code terminal:\n\n"
             f"```\ncd {BASE_DIR}\npython train_model.py\n```")
    st.stop()


st.markdown('<div class="sec-title">👤 Personal Information</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
age            = c1.number_input("Age", 21, 65, 30, 1)
annual_income  = c2.number_input("Annual Income (₹)", 100000, 3000000, 600000, 50000)
monthly_balance= c3.number_input("Monthly Balance (₹)", 0, 200000, 20000, 1000,
                                  help="Savings left after all expenses")


st.markdown('<div class="sec-title">🏦 Credit Accounts</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
num_credit_cards  = c4.slider("Credit Cards", 0, 10, 2)
num_loans         = c5.slider("Active Loans", 0, 7, 1)
credit_history_yrs= c6.slider("Credit History (years)", 0, 30, 5)


st.markdown('<div class="sec-title">💰 Debt & Payments</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
outstanding_debt   = c7.number_input("Outstanding Debt (₹)",  0, 1000000, 50000, 10000)
emi_monthly        = c8.number_input("Monthly EMI Total (₹)", 0, 100000, 8000, 500)

c9, c10 = st.columns(2)
credit_utilization = c9.slider("Credit Utilization (%)", 1, 99, 35,
                                help="% of credit limit you are using. Below 30% is ideal.")
interest_rate      = c10.slider("Interest Rate (%)", 5, 36, 14)


st.markdown('<div class="sec-title">📅 Payment History</div>', unsafe_allow_html=True)
c11, c12 = st.columns(2)
num_delayed_payments = c11.slider("Delayed Payments",  0, 25, 2,
                                   help="How many payments were delayed")
missed_payments      = c12.slider("Missed Payments",   0, 15, 1,
                                   help="Payments that were completely missed")

st.markdown("---")


if st.button("🔍 Check My Credit Score", use_container_width=True, type="primary"):

    # Derived features (same as training)
    debt_to_income = round(outstanding_debt / (annual_income + 1), 3)
    savings_rate   = round(monthly_balance  / ((annual_income / 12) + 1), 3)

    input_vec = np.array([[
        age, annual_income, num_credit_cards, num_loans,
        num_delayed_payments, outstanding_debt, credit_utilization,
        credit_history_yrs, monthly_balance, emi_monthly,
        interest_rate, missed_payments, debt_to_income, savings_rate
    ]])

    input_scaled = scaler.transform(input_vec)
    pred_idx     = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]
    label        = meta["label_map"][pred_idx]   
    confidence   = proba[pred_idx] * 100

   
    base         = {0: 420, 1: 640, 2: 790}[pred_idx]
    numeric_score= int(base + (confidence - 50) * 1.2)
    numeric_score= max(300, min(900, numeric_score))

   
    gauge_pct    = int((numeric_score - 300) / 600 * 100)
    gauge_color  = {"Good": "#10b981", "Standard": "#f59e0b", "Poor": "#ef4444"}[label]

    st.markdown("---")
    st.markdown(f"""
    <div class="meter-wrap">
        <div class="meter-score" style="color:{gauge_color}">{numeric_score}</div>
        <div class="meter-label" style="color:{gauge_color}">Estimated CIBIL-style Score</div>
    </div>
    <div class="gauge-bar-wrap">
        <div class="gauge-bar" style="width:{gauge_pct}%; background:{gauge_color};"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#64748b;margin-bottom:0.5rem">
        <span>300 — Poor</span><span>580 — Standard</span><span>750 — Good</span><span>900</span>
    </div>
    """, unsafe_allow_html=True)

    
    if label == "Good":
        st.markdown(f"""
        <div class="card-good">
            <h2>🟢 GOOD Credit Score</h2>
            <p>Confidence: {confidence:.1f}% &nbsp;·&nbsp; You qualify for premium loans at low interest rates!</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <b>✅ Keep it up!</b>
            • Maintain credit utilization below 30%<br>
            • Never miss an EMI — set auto-pay<br>
            • Keep old credit cards active to boost history<br>
            • Avoid applying for multiple loans together
        </div>""", unsafe_allow_html=True)

    elif label == "Standard":
        st.markdown(f"""
        <div class="card-standard">
            <h2>🟡 STANDARD Credit Score</h2>
            <p>Confidence: {confidence:.1f}% &nbsp;·&nbsp; Loans available but at higher interest rates.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <b>⚡ How to reach GOOD score:</b>
            • Pay all EMIs on time — zero delay<br>
            • Reduce credit utilization to below 30%<br>
            • Clear as much outstanding debt as possible<br>
            • Don't close old credit cards — history matters<br>
            • Avoid taking new loans for 6 months
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="card-poor">
            <h2>🔴 POOR Credit Score</h2>
            <p>Confidence: {confidence:.1f}% &nbsp;·&nbsp; Loan rejection likely. Immediate action needed.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="tip-box">
            <b>🚨 Urgent recovery steps:</b>
            • Clear ALL overdue EMIs immediately<br>
            • Stop applying for new loans or cards<br>
            • Pay off high-interest debt first<br>
            • Set up auto-debit for all payments<br>
            • Get a secured credit card to rebuild trust<br>
            • Check your credit report for errors at CIBIL website<br>
            • It takes 6–12 months of consistent payments to recover
        </div>""", unsafe_allow_html=True)

    
    st.markdown("---")
    st.markdown("**📊 Score Breakdown**")

    labels_order = ["Poor", "Standard", "Good"]
    colors_map   = {"Poor": "#ef4444", "Standard": "#f59e0b", "Good": "#10b981"}
    idx_map      = {v: k for k, v in meta["label_map"].items()}

    for lbl in labels_order:
        p   = proba[idx_map[lbl]] * 100
        col = colors_map[lbl]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <div style="width:80px;font-size:13px;color:#374151;font-weight:500">{lbl}</div>
            <div style="flex:1;background:#e2e8f0;border-radius:999px;height:14px;overflow:hidden">
                <div style="width:{p:.1f}%;height:100%;background:{col};border-radius:999px"></div>
            </div>
            <div style="width:45px;font-size:13px;color:#374151;text-align:right">{p:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    
    st.markdown("---")
    st.markdown("**🔍 Your Key Factors**")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Credit Utilization",  f"{credit_utilization}%",
              "✅ Good" if credit_utilization < 30 else "⚠️ High")
    f2.metric("Delayed Payments",    num_delayed_payments,
              "✅ Good" if num_delayed_payments <= 2 else "❌ High")
    f3.metric("Credit History",      f"{credit_history_yrs} yrs",
              "✅ Good" if credit_history_yrs >= 7 else "⚠️ Short")
    f4.metric("Debt-to-Income",      f"{debt_to_income:.2f}",
              "✅ Good" if debt_to_income < 0.4 else "❌ High")

st.markdown('<div class="footer">⚠️ Educational use only. Not official financial advice.</div>',
            unsafe_allow_html=True)
