# 📊 Telecom Customer Churn Prediction System

## 📋 Project Overview
Customer churn—the rate at which subscribers discontinue their service—poses a major operational and financial risk to subscription-based telecommunications providers. Because acquiring a new customer costs 5 to 25 times more than retaining an existing one, moving from reactive support to **predictive churn forecasting** is critical for protecting recurring revenue and maximizing Customer Lifetime Value (CLV).

This project delivers an end-to-end Machine Learning pipeline that ingests raw customer demographics, account details, and billing patterns to predict account cancellation risks early. By combining domain-specific feature engineering with cost-sensitive classifier benchmarking, the system stratifies customers into risk tiers (**Low**, **Medium**, and **High Risk**), enabling proactive retention workflows.

---

## 🎯 Key Objectives
- **Early Risk Detection:** Develop a supervised classification pipeline prioritizing **Recall** to capture at-risk accounts before cancellation.
- **Behavioral Feature Discovery:** Engineer domain metrics (such as historical spending ratios, price surge indicators, and platform engagement counts) to uncover key flight drivers.
- **Multi-Model Benchmarking:** Train and evaluate five distinct model architectures (**Logistic Regression, Random Forest, XGBoost, Support Vector Machine, and MLP Neural Network**) under identical stratified conditions.
- **Production Deployment:** Package data preprocessing, feature transformations, scaling, and inference into a serialized wrapper (`ChurnPredictorPipeline`) using `joblib` artifacts for real-time scoring.

---

## 🛠️ Dataset Architecture & Features

### Initial Dataset Summary
- **Total Records:** 7,043 customer accounts
- **Initial Schema:** 21 attributes (18 categorical, 3 numerical)
- **Target Feature:** `Churn` (`Yes` / `No` $\rightarrow$ mapped to `1` / `0`)
- **Imbalance Ratio:** 73.5% Retained (`0`) vs. 26.5% Churned (`1`)

### Key Feature Engineering Strategy
Six continuous and categorical features were engineered to capture complex account behaviors:

| Feature Name | Logical / Mathematical Formula | Business Rationale & Behavioral Impact |
| :--- | :--- | :--- |
| **`log_tenure`** | $\log(1 + \text{tenure})$ | Reduces tenure skewness and models diminishing churn risk as longevity grows. |
| **`tenure_cohort`** | `[onboarding, established, loyal, veteran]` | Groups accounts into lifecycle stages ($0\text{--}3$, $4\text{--}12$, $13\text{--}36$, $36+$ months). |
| **`avg_monthly_spend`** | $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ | Calculates long-term historical monthly billing baseline. |
| **`charge_diff_from_avg`** | $\text{MonthlyCharges} - \text{avg\_monthly\_spend}$ | Flags price spikes, recent plan upgrades, or billing friction. |
| **`total_addons_count`** | $\sum \mathbb{I}(\text{Service Addon} == \text{'Yes'})$ | Quantifies product adoption across protective services (`OnlineSecurity`, `TechSupport`, etc.). |
| **`is_long_term_contract`** | $\mathbb{I}(\text{Contract} \in [\text{'One year'}, \text{'Two year'}])$ | Encodes formal contractual commitment. |

---

## 🧠 Methodology, Dataset & EDA Summary
