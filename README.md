**a) Problem Statement:**

The objective of this project is to predict customer attrition (churn) for a telecommunications company. By analyzing customer demographics, account information, and service usage patterns, we aim to identify individuals likely to cancel their subscriptions. This allows the business to implement proactive retention strategies, reducing revenue loss and improving customer lifetime value.

**b) Dataset description:**

- **Source:** Kaggle - Telco Customer Churn Dataset
- **Target Variable:** Churn (Yes/No)
- **Instance Size:** 7,032 rows (after removing nulls from TotalCharges)
- **Feature Size:** 19 features (initially), expanded via One-Hot Encoding to meet the assignment requirement of ≥12 features.
- **Key Features:** Tenure, Monthly Charges, Total Charges, Contract Type (Month-to-month, One year, Two year), Internet Service (DSL, Fiber optic), and Payment Method.

**c) Models Used & Comparison Table**

| ML Model Name |	Accuracy |	AUC |	Precision |	Recall |	F1 |	MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression |	0.7512 |	0.7966 |	0.5263 |	0.6417 |	0.5783 |	0.4083 |
| Decision Tree |	0.6697 |	0.7068 |	0.4332 |	0.7861 |	0.5586 |	0.3657 |
| kNN |	0.7621 |	0.7994 |	0.5558 |	0.5241 |	0.5394 |	0.3795 |
| Naive Bayes |	0.7412 |	0.8041 |	0.5102 |	0.6684 |	0.5787 |	0.4043 |
| Random Forest (Ensemble) |	0.7578 |	0.7849 |	0.5350 |	0.6809 |	0.5992 |	0.4357 |
| XGBoost (Ensemble) |	0.7621 |	0.7946 |	0.5629 |	0.4706 |	0.5126 |	0.3594 |

**d) Observation about models' performances**

 | ML Model Name | Observation about model performance |
 | :--- | :--- |
| Logistic Regression | Balanced Performer. Shows a strong balance between Precision and Recall. It achieved a high AUC (0.7966) and the second-highest MCC, proving it is a reliable baseline for this data. | 
| Decision Tree | High Recall Specialist. While it has the lowest overall Accuracy (66.97%), it successfully identified 78.6% of churners (highest Recall). This indicates it captures churners well but at the cost of many false positives. |  
| kNN | Consistent Accuracy. Tied for the highest Accuracy (76.21%). However, its lower Recall (52.4%) suggests it struggles more than the ensembles to identify all customers who are likely to leave. |   
| Naive Bayes | Stable Probability Model. Performed similarly to Logistic Regression in terms of MCC (0.4043) and AUC (0.8041). It is effective at providing a high Recall (66.8%) while maintaining decent accuracy. |
| Random Forest (Ensemble) | Best Overall Predictor. Achieved the highest MCC (0.4357) and F1 Score (0.5992). It provides the most robust predictions by effectively combining multiple decision trees to reduce error. | 
| XGBoost (Ensemble) | High Precision Model. Tied for the highest Accuracy (76.21%) and achieved the best Precision (56.29%). It is the most "conservative" model, making fewer false churn predictions than the others. | 

