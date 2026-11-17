# Loan Default Prediction

A machine learning project that predicts the probability of loan default using LendingClub data (2007–2018). Five models are compared. The deployed model is a tuned, calibrated LightGBM with a cost-weighted decision threshold. An interactive Streamlit dashboard explains each prediction with SHAP.

## Project Structure

```
├── lending_club_models.ipynb   # EDA, feature engineering, training, evaluation, artifact export
├── app.py                      # Streamlit dashboard for real-time default-risk prediction
├── model_artifacts.joblib      # Trained LightGBM + isotonic calibrator + encoders + threshold
├── requirements.txt            # Python dependencies
├── eda_*.png                   # Exploratory data analysis plots
├── evaluation_*.png            # Per-model ROC / PR / confusion-matrix plots
├── calibration_curve.png       # Reliability curve after isotonic calibration
├── cost_threshold_analysis.png # Expected cost vs. decision threshold
├── base_rate_shift_diagnosis.png
├── lgbm_feature_importance.png
└── shap_*.png                  # Global and local SHAP explanations
```

## Pipeline

1. **Data:** the notebook loads the LendingClub accepted-loans file and keeps only resolved loans (Fully Paid or Charged Off/Default). It drops leaky post-origination columns such as `recoveries`. It then takes the 500,000 most recent loans.
2. **Right-censoring check:** recently issued loans have not had time to default, so default rates fall at the end of the data window (see `base_rate_shift_diagnosis.png`). The notebook only uses loans whose full term has elapsed by the data cutoff.
3. **Feature engineering:** 33 features in total. The engineered ones include the income-to-loan, payment-to-income, credit-utilization and open-to-total-accounts ratios, credit history length, total payment, and the `util × DTI` and `rate × DTI` interactions.
4. **Time-based split:** train, validation and test sets are split in date order (70 / 15 / 15), not randomly. Hyperparameter search uses `TimeSeriesSplit` cross-validation.
5. **Models:** Logistic Regression, Decision Tree and Random Forest are tuned with randomized search. LightGBM is tuned with Optuna (30 trials, GPU, early stopping). CatBoost is also trained on GPU.
6. **Calibration:** isotonic regression, fitted on the validation set.
7. **Threshold:** the decision threshold minimizes expected cost, assuming a missed default costs 5× a wrongly rejected good loan.
8. **Explainability:** SHAP summary, bar and waterfall plots.

## Results (held-out test set, 75,000 loans)

| Model | ROC-AUC | PR-AUC | Brier | KS |
|---|---|---|---|---|
| CatBoost | 0.7167 | 0.3368 | 0.2088 | 0.3192 |
| **LightGBM (Tuned)** | 0.7158 | **0.3373** | 0.2057 | 0.3141 |
| Random Forest | 0.7105 | 0.3218 | 0.1959 | 0.3058 |
| Logistic Regression | 0.7056 | 0.3149 | 0.2089 | 0.2992 |
| Decision Tree | 0.6911 | 0.2967 | 0.2234 | 0.2834 |

The test-set default rate is 17.1%, which is the PR-AUC of a random model.

**Deployed model: calibrated LightGBM**

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.7156 (95% bootstrap CI: 0.7106–0.7202) |
| Cost-optimal threshold | 0.1655 (5:1 cost ratio) |
| Default recall at threshold | 0.79 |
| Default precision at threshold | 0.25 |
| F1-optimal threshold (for reference) | 0.2353 (F1 = 0.475) |

LightGBM was deployed instead of CatBoost: their AUCs are within noise of each other, and LightGBM had the slightly higher PR-AUC, the better Brier score and faster inference. The threshold deliberately favors catching defaults (high recall) over precision, because a missed default costs far more than a rejected good loan.

## Dataset

The raw data is not included in this repository due to file size. Download from Kaggle:

- [LendingClub Accepted Loans 2007–2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Place `accepted_2007_to_2018Q4.csv.gz` in a `dataset/` folder before running the notebook.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/am4nsingh/Loan-Default-Prediction.git
cd Loan-Default-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain — run the notebook end-to-end; it regenerates model_artifacts.joblib
jupyter notebook lending_club_models.ipynb

# 4. Launch the dashboard (works out of the box with the committed model_artifacts.joblib)
streamlit run app.py
```

## Dashboard

Enter the applicant's details in the sidebar: sub-grade, interest rate, loan amount, installment, term, income, DTI, employment length, home ownership, purpose, credit history, revolving utilization and recent inquiries. The app then shows:

- the calibrated probability of default,
- an approve or reject recommendation, based on the saved cost-optimal threshold,
- a SHAP waterfall plot of the factors driving that prediction.

## Tech Stack

- Python, Pandas, NumPy
- scikit-learn, LightGBM, CatBoost, Optuna
- SHAP
- Matplotlib, Seaborn
- Streamlit
- Joblib

## Author

**Aman Singh** — [github.com/am4nsingh](https://github.com/am4nsingh)
