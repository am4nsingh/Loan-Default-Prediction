import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import traceback
import os

# --- 1. Load All Artifacts (Cached for speed) ---

@st.cache_resource
def load_artifacts():
    """Loads model artifacts, falling back to individual files if the bundle fails."""
    model = iso_cal = imputation_values = None
    categorical_features = feature_columns = label_encoders = None
    optimal_threshold = 0.3

    def _load_individual():
        m = joblib.load('lgbm_model.joblib')
        ic = joblib.load('iso_calibrator.joblib') if os.path.exists('iso_calibrator.joblib') else None
        iv = joblib.load('imputation_values.joblib')
        cf = joblib.load('categorical_features.joblib')
        fc = joblib.load('feature_columns.joblib')
        le = joblib.load('label_encoders.joblib')
        return m, ic, iv, cf, fc, le

    try:
        if os.path.exists('model_artifacts.joblib'):
            try:
                artifacts = joblib.load('model_artifacts.joblib')
                model = artifacts['model']
                iso_cal = artifacts.get('iso_calibrator')
                imputation_values = artifacts['imputation_values']
                categorical_features = artifacts['categorical_features']
                feature_columns = artifacts['feature_columns']
                label_encoders = artifacts['label_encoders']
                optimal_threshold = artifacts.get('optimal_threshold', 0.3)
            except Exception:
                model, iso_cal, imputation_values, categorical_features, feature_columns, label_encoders = _load_individual()
        else:
            model, iso_cal, imputation_values, categorical_features, feature_columns, label_encoders = _load_individual()

    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.code(traceback.format_exc())
        return (None,) * 7

    return model, iso_cal, imputation_values, categorical_features, feature_columns, label_encoders, optimal_threshold

# Load artifacts
model, ISO_CALIBRATOR, IMPUTATION_VALUES, CAT_FEATURES, FEATURE_COLS, LABEL_ENCODERS, OPTIMAL_THRESHOLD = load_artifacts()

all_artifacts_loaded = all(v is not None for v in [model, IMPUTATION_VALUES, CAT_FEATURES, FEATURE_COLS, LABEL_ENCODERS])

# --- 2. App Title and Sidebar ---
st.set_page_config(layout="wide")
st.title("Loan Default Risk Dashboard 🏦")

# --- 3. Main App Logic ---
if all_artifacts_loaded:
    st.write("Enter applicant details in the sidebar to get a real-time risk prediction and explanation.")
    st.sidebar.header("Applicant Information")

    # --- User Input Widgets ---
    SUB_GRADES = [f"{g}{n}" for g in 'ABCDEFG' for n in range(1, 6)]
    sub_grade = st.sidebar.selectbox("Sub-Grade", SUB_GRADES, index=SUB_GRADES.index('C1'))
    int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", 500, 40000, 10000, 100)
    installment = st.sidebar.number_input("Monthly Installment ($)", 10.0, 1500.0, 300.0, 10.0)
    term = st.sidebar.select_slider("Loan Term", ["36 months", "60 months"])
    annual_inc = st.sidebar.number_input("Annual Income ($)", 10000, 2000000, 75000, 100)
    dti = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 15.0, 0.1)
    emp_length = st.sidebar.selectbox("Employment Length", ['10+ years', '2 years', '3 years', '< 1 year', '1 year',
                                                           '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', 'n/a'])
    home_ownership = st.sidebar.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'ANY'])
    purpose = st.sidebar.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business', 'car', 'medical', 'house', 'vacation'])
    credit_history_days = st.sidebar.number_input("Credit History (Days)", 300, 10000, 4500, 10)
    revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 50.0, 0.1)
    inq_last_6mths = st.sidebar.number_input("Inquiries in Last 6 Months", 0, 10, 1, 1)

    # --- 4. Preprocessing and Prediction Function ---
    def predict_default_risk(input_data):
        # 1. Start from imputation defaults, overlay with user inputs
        data_dict = IMPUTATION_VALUES.copy()
        data_dict.update(input_data)

        # 2. Compute engineered features from RAW values first
        data_dict['income_to_loan_ratio'] = (
            data_dict.get('annual_inc', 0) / (data_dict.get('loan_amnt', 0) + 1)
        )
        data_dict['credit_utilization_ratio'] = (
            data_dict.get('revol_bal', 0) / (data_dict.get('total_acc', 0) + 1)
        )

        # FIX: Normalize term string — strip leading/trailing spaces
        term_str = str(data_dict.get('term', '36 months')).strip()
        data_dict['term'] = term_str
        data_dict['term_in_months'] = float(term_str.replace(' months', ''))

        data_dict['total_payment'] = (
            data_dict['term_in_months'] * data_dict.get('installment', 0)
        )
        data_dict['payment_to_income_ratio'] = (
            data_dict.get('installment', 0) / (data_dict.get('annual_inc', 0) + 1)
        )
        data_dict['open_acc_to_total_acc'] = (
            data_dict.get('open_acc', 0) / (data_dict.get('total_acc', 0) + 1)
        )
        data_dict['util_x_dti'] = (
            data_dict.get('revol_util', 0) * data_dict.get('dti', 0)
        )
        data_dict['rate_x_dti'] = (
            data_dict.get('int_rate', 0) * data_dict.get('dti', 0)
        )
        data_dict['installment_to_loan'] = (
            data_dict.get('installment', 0) / (data_dict.get('loan_amnt', 0) + 1)
        )

        # 3. THEN create DataFrame in correct column order
        df = pd.DataFrame([data_dict])[FEATURE_COLS]

        # 4. Encode Categorical Features — keep as int, not pandas 'category'
        df_processed = df.copy()
        for col in CAT_FEATURES:
            if col in df_processed.columns:
                le = LABEL_ENCODERS[col]
                val = str(df_processed[col].iloc[0])
                if val in le.classes_:
                    df_processed[col] = le.transform([val])[0]
                else:
                    df_processed[col] = -1
                df_processed[col] = df_processed[col].astype('int32')

        # 5. Predict — raw proba then calibrate with isotonic regression
        raw_proba = model.predict_proba(df_processed)[:, 1][0]
        if ISO_CALIBRATOR is not None:
            calibrated_proba = float(ISO_CALIBRATOR.transform([raw_proba])[0])
        else:
            calibrated_proba = raw_proba
        return calibrated_proba, df_processed

    # --- 5. Display Prediction and Explanation ---
    if st.sidebar.button("Calculate Default Risk"):
        user_input = {
            'sub_grade': sub_grade,
            'int_rate': int_rate,
            'loan_amnt': loan_amnt,
            'installment': installment,
            'term': term,
            'annual_inc': annual_inc,
            'dti': dti,
            'emp_length': emp_length,
            'home_ownership': home_ownership,
            'purpose': purpose,
            'credit_history_days': credit_history_days,
            'revol_util': revol_util,
            'inq_last_6mths': inq_last_6mths
        }

        prob, processed_df = predict_default_risk(user_input)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.header("Risk Profile")
            st.metric("Probability of Default", f"{prob:.2%}")

            if prob > OPTIMAL_THRESHOLD:
                st.error("High Risk (Above Threshold)")
                st.write(f"This applicant's risk ({prob:.2%}) is above the {OPTIMAL_THRESHOLD:.2%} threshold. **Rejection Recommended.**")
            else:
                st.success("Low Risk (Below Threshold)")
                st.write(f"This applicant's risk ({prob:.2%}) is below the {OPTIMAL_THRESHOLD:.2%} threshold. **Approval Recommended.**")

            st.subheader("Key Applicant Inputs")
            st.json(user_input)

        with col2:
            st.header("Top Risk Factors (SHAP Explanation)")

            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            shap_values_list = explainer.shap_values(processed_df)

            if isinstance(explainer.expected_value, (list, np.ndarray)):
                expected_value = explainer.expected_value[1]
            else:
                expected_value = explainer.expected_value

            if isinstance(shap_values_list, list) and len(shap_values_list) == 2:
                shap_values_class_1 = shap_values_list[1]
            else:
                shap_values_class_1 = shap_values_list

            shap_explanation = shap.Explanation(
                values=shap_values_class_1[0, :],
                base_values=expected_value,
                data=processed_df.iloc[0, :],
                feature_names=processed_df.columns.tolist()
            )

            fig, ax = plt.subplots(figsize=(10, 10))
            shap.waterfall_plot(shap_explanation, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("Please fill in the applicant details on the left and click 'Calculate Default Risk'.")

else:
    st.error("Application could not load model artifacts. Please check the file paths and ensure all .joblib files are in the same directory as app.py.")
