# Bank Customer Churn Prediction

## Business Problem

Banks lose significant revenue and acquisition costs when customers close accounts or move to competitors.  
Early identification of at-risk customers allows proactive retention efforts, personalized offers, and reduced churn impact on profitability.

## Objective

- Predict the probability of a customer churning (closing their account) in the near term  
- Identify and rank the most influential churn drivers  
- Provide actionable insights and risk scores to support retention and marketing teams

## Data Description

- Synthetic dataset modeled 
- ~18,000 records  
- Structured tabular data with ~10–12 features (tenure, balance, credit score proxies, product count, complaints, etc.)

## Approach / What I Did

- Generated realistic synthetic data  
- Performed exploratory data analysis to uncover churn patterns by segment  
- Conducted feature engineering (ratios, binary flags, interactions)  
- Trained and compared baseline heuristic + multiple models (Logistic Regression, Random Forest, XGBoost)  
- Handled class imbalance using SMOTE and class weights  
- Evaluated using ROC-AUC, Precision, Recall, F1; included feature importance and SHAP explanations  
- Simulated retention impact by targeting high-risk customers

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib/Seaborn, MLflow (tracking), PySpark (planned scaling), Power BI (dashboard prototype)

## Key Results / Insights

- Achieved test ROC-AUC of ~0.86–0.88 (0.12–0.16 improvement over simple rule-based baseline)  
- Top churn drivers: short tenure (median churner tenure ~11 months vs 25 for retained), month-to-month-like behavior (31% churn rate vs ~1% for longer commitments), high support tickets/complaints  
- Churn class recall ~70–78% at reasonable precision (~60–70%)  
- High-risk segment (top 15–20%) shows 3–5× higher churn probability than average

## Business Impact / How This Is Used

Risk scores and driver insights enable retention teams to prioritize outreach to the top 15–20% highest-risk customers.  
Simulated intervention (targeted offers/discounts) projected 10–14% relative reduction in expected churn volume (assuming 45–55% success rate in saving at-risk accounts).  
Dashboard visualizes segments, trends, and priority lists for quick action by marketing and relationship managers.