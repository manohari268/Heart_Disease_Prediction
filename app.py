# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
# -------------------------
# Page config & Theme 
# -------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide", page_icon="❤")
PRIMARY = "#0b62d6"
ACCENT = "#2b9af3"
CARD_BG = "#f6fbff"
def local_css():
    st.markdown(
        f"""
        <style>
            .big-font {{ font-size:26px; font-weight:600; color:{PRIMARY}; }}
            .card {{
                background: {CARD_BG};
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 3px 8px rgba(11,98,214,0.06);
            }}
            .muted {{ color:#6b7280; font-size:13px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
local_css()
# -------------------------
# Utilities
# -------------------------
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
@st.cache_resource
def load_csv(path="framingham.csv"):
    try:
        df = pd.read_csv(path)
        df = df.fillna(df.mean(numeric_only=True))
        return df
    except Exception:
        return None
def create_pdf_bytes(pred_label, percent, inputs_dict):
    """Create a simple PDF report and return bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.92, "Heart Disease Prediction Report", ha="center", fontsize=18, weight="bold", color=PRIMARY)
        ax.text(0.08, 0.86, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=9)
        ax.text(0.08, 0.80, f"Prediction: {pred_label}", fontsize=12)
        ax.text(0.08, 0.76, f"Risk probability: {percent:.2f}%", fontsize=11)
        ax.text(0.08, 0.72, "Input values:", fontsize=12, weight="bold")
        y = 0.70
        for k, v in inputs_dict.items():
            ax.text(0.09, y, f"- {k}: {v}", fontsize=9)
            y -= 0.02
            if y < 0.12:
                break
        pdf.savefig(fig)
        plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
# -------------------------
# Load resources
# -------------------------
df = load_csv("framingham.csv")
scaler = load_pickle("scaler.pkl")
models = {
    "Logistic Regression": load_pickle("lr_model.pkl"),
    "SVM": load_pickle("svm_model.pkl"),
    "Random Forest": load_pickle("heart_model.pkl"),
}
ewoa_params = load_pickle("ewoa_best_params.pkl")
ewoa_history = load_pickle("ewoa_history.pkl")
# Fixed FEATURES list (must match training)
FEATURES = [
    'male','age','education','currentSmoker','cigsPerDay',
    'BPMeds','prevalentStroke','prevalentHyp','diabetes',
    'totChol','sysBP','diaBP','BMI','heartRate','glucose'
]
# -------------------------
# Authentication (simple, local)
# -------------------------
USERNAME = "admin"
PASSWORD = "heart123"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
def do_login(u, p):
    if u == USERNAME and p == PASSWORD:
        st.session_state.logged_in = True
        st.success("Login successful")
        try:
            st.rerun()
        except Exception:
            pass
    else:
        st.error("Invalid username or password")
def do_logout():
    st.session_state.logged_in = False
    try:
        st.rerun()
    except Exception:
        pass
# -------------------------
# Header
# -------------------------
st.markdown(
    f"""<div style="background:{PRIMARY};padding:12px;border-radius:8px;margin-bottom:12px">
            <h2 style="color:white;margin:0 8px">Heart Disease Prediction</h2>
        </div>""",
    unsafe_allow_html=True,
)
# -------------------------
# Login screen
# -------------------------
if not st.session_state.logged_in:
    st.markdown("<div class='card'><h3>Admin Login</h3></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        uname = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            do_login(uname.strip(), pwd.strip())
    st.stop()
# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.selectbox("Navigation", [ "Prediction", "Dashboard" ,"Logout"])
if page == "Logout":
    do_logout()
# -------------------------
# Compute model accuracies (cached)
# -------------------------
if "acc_dict" not in st.session_state:
    if df is not None and scaler is not None:
        try:
            X_full = df[FEATURES]
            y_full = df["TenYearCHD"]
            X_full_scaled = scaler.transform(X_full)
            accd = {}
            for name, mdl in models.items():
                if mdl is None:
                    accd[name] = 0.0
                else:
                    try:
                        preds = mdl.predict(X_full_scaled)
                        accd[name] = float(accuracy_score(y_full, preds))
                    except Exception:
                        accd[name] = 0.0
            st.session_state.acc_dict = accd
        except Exception:
            st.session_state.acc_dict = {}
    else:
        st.session_state.acc_dict = {}
# -------------------------
# Pages
# -------------------------
# =========================================
# PREDICTION PAGE  
# =========================================
if page == "Prediction":
    st.write("Enter patient information below:")
    X_cols = FEATURES
    # ---------------- BMI CALCULATOR ----------------
    st.markdown(" BMI Calculator ")
    with st.expander("Calculate BMI"):
        w = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
        h = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
        if st.button("Calculate BMI"):
            bmi_calc = w / ((h / 100) ** 2)
            st.success(f"Calculated BMI = *{bmi_calc:.2f}*")
            st.session_state.calculated_bmi = bmi_calc
        else:
            st.session_state.calculated_bmi = None
    st.markdown("---")
    # ---------------- INPUT FORM ----------------
    with st.form("predict_form"):
        inputs = {}
        col1, col2 = st.columns(2)
        for i, cname in enumerate(X_cols):
            minv = float(df[cname].min())
            maxv = float(df[cname].max())
            meanv = float(df[cname].mean())
            help_text = f"Range: {minv} → {maxv} | Average: {meanv:.2f}"
            # If BMI and a calculated BMI exists → auto fill
            if cname == "BMI" and st.session_state.calculated_bmi is not None:
                default_val = float(st.session_state.calculated_bmi)
            else:
                default_val = meanv
            # Left / right columns
            if i % 2 == 0:
                inputs[cname] = col1.number_input(
                    f"{cname}", 
                    min_value=minv, max_value=maxv, value=default_val,
                    help=help_text
                )
            else:
                inputs[cname] = col2.number_input(
                    f"{cname}", 
                    min_value=minv, max_value=maxv, value=default_val,
                    help=help_text
                )
        submit = st.form_submit_button("Predict")
    # ---------------- ON SUBMIT ----------------
    if submit:
        # Convert to DF
        input_df = pd.DataFrame([inputs])[X_cols]
        try:
            X_scaled = scaler.transform(input_df)
        except:
            st.error("❌ Error preparing input for model. Check scaler and features.")
            st.stop()
        # --------------------------------
        # MODEL PROBABILITIES
        # --------------------------------
        probs = {}
        for name, mdl in models.items():
            if mdl is None:
                probs[name] = 0.0
                continue
            try:
                if hasattr(mdl, "predict_proba"):
                    probs[name] = float(mdl.predict_proba(X_scaled)[:, 1][0])
                else:
                    probs[name] = float(mdl.predict(X_scaled)[0])
            except:
                probs[name] = 0.0
        avg_prob = np.mean(list(probs.values()))
        risk_label = "High Risk" if avg_prob >= 0.5 else "Low Risk"
        # --------------------------------
        # PREDICTION OUTPUT
        # --------------------------------
        st.markdown(f"## Prediction Result: *{risk_label}*")
        st.metric("Risk Score (%)", f"{avg_prob*100:.2f}")
        # Popup messages
        if risk_label == "Low Risk":
            st.success("🟢 Your risk is LOW. Continue maintaining a healthy lifestyle!")
        else:
            st.error("🔴 HIGH RISK detected! It is strongly advised to consult a cardiologist.")
        # PDF Download
        pdf_bytes = create_pdf_bytes(risk_label, avg_prob*100, inputs)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="heart_prediction_report.pdf",
            mime="application/pdf"
        )
# =========================================
# DASHBOARD  (Compressed + Clean)
# =========================================
elif page == "Dashboard":
    st.subheader("Feature Contribution → Heart Disease Risk")
    # ====== SMALL FEATURE-WISE GRAPHS ======
    numeric_feats = df[FEATURES].select_dtypes(include=[np.number]).columns.tolist()
    if "TenYearCHD" in numeric_feats:
        numeric_feats.remove("TenYearCHD")
    @st.cache_data
    def build_feature_risk(df_local, feats, target="TenYearCHD", bins=6):
        agg = {}
        for f in feats:
            uniq = df_local[f].nunique()
            if uniq <= 4:
                grp = df_local.groupby(f)[target].mean() * 100
                agg[f] = ("cat", grp)
            else:
                try:
                    binned, edges = pd.qcut(df_local[f], q=bins, retbins=True, duplicates="drop")
                    tmp = df_local[[f, target]].copy()
                    tmp["bin"] = pd.cut(tmp[f], bins=edges, include_lowest=True)
                    grp = tmp.groupby("bin")[target].mean() * 100
                    labels = [f"{round(iv.left)}-{round(iv.right)}" for iv in grp.index]
                    agg[f] = ("num", (labels, grp.values))
                except:
                    tmp = df_local[[f, target]]
                    tmp["r"] = tmp[f].round()
                    grp = tmp.groupby("r")[target].mean() * 100
                    agg[f] = ("num_simple", (grp.index.astype(str), grp.values))
        return agg
    feat_agg = build_feature_risk(df, numeric_feats)
    cols_per_row = 3
    flist = list(feat_agg.keys())
    for i in range(0, len(flist), cols_per_row):
        row_feats = flist[i:i+cols_per_row]
        cols = st.columns(len(row_feats))
        for col, fname in zip(cols, row_feats):
            ttype, data = feat_agg[fname]
            if ttype == "cat":
                fig = px.bar(
                    x=data.index.astype(str),
                    y=data.values,
                    title=fname,
                    labels={"x": fname, "y": "Risk (%)"},
                    template="plotly_white"
                )
            else:
                x, y = data
                fig = px.line(
                    x=x, y=y, markers=True,
                    title=fname,
                    labels={"x": fname, "y": "Risk (%)"},
                    template="plotly_white"
                )

            fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
            col.plotly_chart(fig, width="stretch")
    st.markdown("---")
    # ==============================
    # CONFUSION MATRIX
    # ==============================
    st.subheader("Confusion Matrix ")
    from sklearn.metrics import confusion_matrix
    Xf = df[FEATURES]
    yf = df["TenYearCHD"]
    Xfs = scaler.transform(Xf)
    rf = models["Random Forest"]
    ypred = rf.predict(Xfs)
    cm = confusion_matrix(yf, ypred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"],
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    fig_cm.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig_cm, width="stretch")
    st.markdown("---")
    # ------------------ SUMMARY METRICS ------------------
    col1, col2, col3, col4 = st.columns(4)
    alive_percentage = (1 - df["TenYearCHD"].mean()) * 100
    avg_age = df["age"].mean()
    total_alive = (df["TenYearCHD"] == 0).sum()
    total_cases = (df["TenYearCHD"] == 1).sum()
    col1.metric("Survival %", f"{alive_percentage:.1f}%")
    col2.metric("Avg Age", f"{avg_age:.1f}")
    col3.metric("No Disease", total_alive)
    col4.metric("Heart Disease", total_cases)
    st.markdown("---")
    # ==============================
    # MODEL ACCURACY
    # ==============================
    st.subheader("Model Accuracy")
    acc_df = pd.DataFrame.from_dict(st.session_state.acc_dict,
                                    orient="index", columns=["Accuracy"])
    acc_df["Accuracy %"] = acc_df["Accuracy"] * 100
    st.table(acc_df.style.format({"Accuracy %": "{:.2f}%"}))
    fig_acc = px.bar(
        x=list(st.session_state.acc_dict.keys()),
        y=list(st.session_state.acc_dict.values()),
        text=[f"{v*100:.2f}%" for v in st.session_state.acc_dict.values()],
        labels={"x": "Model", "y": "Accuracy"},
        title="Model Accuracy Comparison",
        template="plotly_white"
    )
    fig_acc.update_traces(textposition="outside")
    fig_acc.update_layout(height=300)
    st.plotly_chart(fig_acc, width="stretch")      