import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

# ── Load saved artifacts ─────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = pickle.load(open("models/gb_model.pkl", "rb"))
    scaler       = pickle.load(open("models/scaler.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ── Shared preprocessing (used by both tabs) ─────────────
def preprocess_df(df):
    df = df.copy()

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Drop customerID if present
    customer_ids = df['customerID'].copy() if 'customerID' in df.columns else None
    df.drop(columns=['customerID'], errors='ignore', inplace=True)

    # Drop Churn column if present (in case user uploads full dataset)
    df.drop(columns=['Churn'], errors='ignore', inplace=True)

    # Binary encode
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map(binary_map)

    # One-hot encode
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                  'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Align to training columns
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scale numerics
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])

    return df, customer_ids

# ── Single prediction feature builder ────────────────────
def build_single_input(gender, senior, partner, dependents, tenure,
                        phone_service, multiple_lines, internet,
                        online_sec, online_backup, device_prot,
                        tech_support, streaming_tv, streaming_movies,
                        contract, paperless, payment,
                        monthly_charges, total_charges):
    row = {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "PhoneService":     1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    row["MultipleLines_No phone service"] = 1 if multiple_lines == "No phone service" else 0
    row["MultipleLines_Yes"]              = 1 if multiple_lines == "Yes" else 0
    row["InternetService_Fiber optic"]    = 1 if internet == "Fiber optic" else 0
    row["InternetService_No"]             = 1 if internet == "No" else 0

    for col, val in [("OnlineSecurity", online_sec), ("OnlineBackup", online_backup),
                     ("DeviceProtection", device_prot), ("TechSupport", tech_support),
                     ("StreamingTV", streaming_tv), ("StreamingMovies", streaming_movies)]:
        row[f"{col}_No internet service"] = 1 if val == "No internet service" else 0
        row[f"{col}_Yes"]                 = 1 if val == "Yes" else 0

    row["Contract_One year"]                     = 1 if contract == "One year" else 0
    row["Contract_Two year"]                     = 1 if contract == "Two year" else 0
    row["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"]        = 1 if payment == "Electronic check" else 0
    row["PaymentMethod_Mailed check"]            = 1 if payment == "Mailed check" else 0

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols, fill_value=0)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# ── Risk signals helper ───────────────────────────────────
def get_risk_signals(row):
    signals = []
    if str(row.get('Contract', '')).strip() == 'Month-to-month':
        signals.append("Month-to-month contract")
    if str(row.get('InternetService', '')).strip() == 'Fiber optic':
        signals.append("Fiber optic service")
    if float(row.get('tenure', 99)) < 12:
        signals.append("Low tenure (<12 months)")
    if float(row.get('MonthlyCharges', 0)) > 5000:
        signals.append("High monthly charges")
    if str(row.get('PaymentMethod', '')).strip() == 'Electronic check':
        signals.append("Electronic check payment")
    if str(row.get('OnlineSecurity', '')).strip() == 'No':
        signals.append("No online security")
    return ", ".join(signals) if signals else "None"

# ═════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════
st.title("📉 Customer Churn Predictor")
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])


# ─────────────────────────────────────────────────────────
# TAB 1 — Single Prediction
# ─────────────────────────────────────────────────────────
with tab1:
    st.markdown("Fill in the customer details on the left and hit **Predict**.")
    st.sidebar.header("👤 Customer Details")

    gender           = st.sidebar.selectbox("Gender",              ["Male", "Female"])
    senior           = st.sidebar.selectbox("Senior Citizen",      ["No", "Yes"])
    partner          = st.sidebar.selectbox("Has Partner",         ["Yes", "No"])
    dependents       = st.sidebar.selectbox("Has Dependents",      ["Yes", "No"])
    tenure           = st.sidebar.slider("Tenure (months)",        0, 72, 12)
    phone_service    = st.sidebar.selectbox("Phone Service",       ["Yes", "No"])
    multiple_lines   = st.sidebar.selectbox("Multiple Lines",      ["No", "Yes", "No phone service"])
    internet         = st.sidebar.selectbox("Internet Service",    ["DSL", "Fiber optic", "No"])
    online_sec       = st.sidebar.selectbox("Online Security",     ["Yes", "No", "No internet service"])
    online_backup    = st.sidebar.selectbox("Online Backup",       ["Yes", "No", "No internet service"])
    device_prot      = st.sidebar.selectbox("Device Protection",   ["Yes", "No", "No internet service"])
    tech_support     = st.sidebar.selectbox("Tech Support",        ["Yes", "No", "No internet service"])
    streaming_tv     = st.sidebar.selectbox("Streaming TV",        ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies",    ["Yes", "No", "No internet service"])
    contract         = st.sidebar.selectbox("Contract Type",       ["Month-to-month", "One year", "Two year"])
    paperless        = st.sidebar.selectbox("Paperless Billing",   ["Yes", "No"])
    payment          = st.sidebar.selectbox("Payment Method",      [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=18.0,
    max_value=120.0,
    value=70.0,
    step=1.0
)
    total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=9000.0,
    value=round(monthly_charges * tenure, 2),
    step=10.0
)

    if st.button("🔍 Predict Churn", use_container_width=True):
        input_df = build_single_input(
            gender, senior, partner, dependents, tenure,
            phone_service, multiple_lines, internet,
            online_sec, online_backup, device_prot,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless, payment,
            monthly_charges, total_charges
        )
        prob = model.predict_proba(input_df)[0][1]
        pred = prob >= 0.40

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with col2:
            if pred:
                st.error("⚠️ HIGH RISK — Likely to Churn")
            else:
                st.success("✅ LOW RISK — Likely to Stay")
        with col3:
            risk = "🔴 High" if prob > 0.6 else ("🟡 Medium" if prob > 0.4 else "🟢 Low")
            st.metric("Risk Level", risk)

        st.markdown("---")
        st.markdown("### Churn Probability Breakdown")
        st.progress(float(prob))
        st.caption(f"Stay: {(1-prob)*100:.1f}%  |  Churn: {prob*100:.1f}%")

        st.markdown("### 💡 Key Risk Signals")
        raw_row = {
            'Contract': contract, 'InternetService': internet,
            'tenure': tenure, 'MonthlyCharges': monthly_charges,
            'PaymentMethod': payment, 'OnlineSecurity': online_sec
        }
        signals = get_risk_signals(raw_row).split(", ")
        if signals and signals[0] != "None":
            for s in signals:
                st.warning(f"📌 {s}")
        else:
            st.info("No major risk signals detected.")


# ─────────────────────────────────────────────────────────
# TAB 2 — Batch Prediction
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📂 Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV with the same columns as the Telco dataset. `customerID` and `Churn` columns are optional.")

    # ── Sample CSV download ───────────────────────────────
    sample = pd.DataFrame([{
        "customerID": "1234-SAMPLE",
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0, "TotalCharges": "475.0"
    }])
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=sample.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        raw_df = pd.read_csv(uploaded)

        st.markdown(f"**{len(raw_df)} customers loaded.** Preview:")
        st.dataframe(raw_df.head(), use_container_width=True)

        # ── ONLY RUN IF BUTTON CLICKED ────────────────────────
        if st.button("⚡ Run Batch Prediction", use_container_width=True):
            with st.spinner("Predicting..."):
                processed, customer_ids = preprocess_df(raw_df)

                probs = model.predict_proba(processed)[:, 1]
                preds = (probs >= 0.40).astype(int)

                # Build results from raw_df (same length as processed)
                results = pd.DataFrame({'customerID': customer_ids if customer_ids is not None else [f"Row {i+1}" for i in range(len(processed))]})

                results['Churn Probability (%)'] = (probs * 100).round(1)
                results['Prediction']            = np.where(preds == 1, '⚠️ Churn', '✅ Stay')
                results['Risk Level']            = pd.cut(
                    probs,
                    bins=[0, 0.4, 0.6, 1.0],
                    labels=['🟢 Low', '🟡 Medium', '🔴 High']
                )
                results['Risk Signals'] = raw_df.apply(get_risk_signals, axis=1)

                # Save everything to session_state so it survives reruns!
                st.session_state['batch_results'] = results
                st.session_state['batch_probs'] = probs
                st.session_state['batch_preds'] = preds

        # ── DISPLAY SAVED STATE IF IT EXISTS ──────────────────
        if 'batch_results' in st.session_state:
            results = st.session_state['batch_results']
            probs = st.session_state['batch_probs']
            preds = st.session_state['batch_preds']
            
            st.markdown("---")
            st.markdown("### 📊 Summary")
            total     = len(results)
            churners  = preds.sum()
            stay      = total - churners
            avg_prob  = probs.mean() * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers", total)
            m2.metric("Predicted to Churn", f"{churners} ({churners/total*100:.1f}%)")
            m3.metric("Predicted to Stay",  f"{stay} ({stay/total*100:.1f}%)")
            m4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

            # ── Risk breakdown ────────────────────────────
            st.markdown("### 🗂️ Risk Breakdown")
            risk_counts = results['Risk Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            st.dataframe(risk_counts, use_container_width=True)

            # ── Full results table ────────────────────────
            st.markdown("### 📋 Full Results")

            # Filter controls
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                risk_filter = st.multiselect(
                    "Filter by Risk Level",
                    ['🟢 Low', '🟡 Medium', '🔴 High'],
                    default=['🔴 High', '🟡 Medium']
                )
            with filter_col2:
                sort_by = st.selectbox("Sort by", ["Churn Probability (%) ↓", "Churn Probability (%) ↑"])

            filtered = results[results['Risk Level'].isin(risk_filter)] if risk_filter else results
            ascending = sort_by.endswith("↑")
            filtered  = filtered.sort_values("Churn Probability (%)", ascending=ascending)

            st.dataframe(filtered, use_container_width=True, height=400)

            # ── Download results ──────────────────────────
            st.markdown("---")
            csv_out = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )