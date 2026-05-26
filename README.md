# 📉 Telco Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Analysis](https://img.shields.io/badge/Data%20Analysis-Pandas%20%7C%20NumPy-lightgrey)

## 📌 Project Overview
Customer retention is a primary driver of profitability in the telecom industry. In highly competitive markets, acquiring a new customer is significantly more expensive than retaining an existing one. 

This project provides an end-to-end machine learning analysis of telecom customer churn. By analyzing a dataset of **7,032 customers** (with a baseline churn rate of **26.6%**), this project identifies key behavioral drivers of churn and develops a predictive model to flag at-risk customers. 

**Skills Demonstrated:**
*   **Business Acumen:** Framing data science problems around business KPIs (optimizing for Recall to minimize false negatives).
*   **Data Processing & EDA:** Handling missing values, encoding multi-class features, and statistical visualizations.
*   **Feature Engineering:** Creating custom engagement and financial volatility metrics.
*   **Predictive Modeling:** Training, evaluating, and interpreting Logistic Regression and Random Forest classifiers.
*   **Actionable Strategy:** Translating model feature importances into tangible marketing campaigns.

## 🏢 Business Problem
**What is Churn?** In this context, churn refers to a customer terminating their active subscription with the telecom provider. 

**Why measure and reduce it?** Failing to proactively identify churn results in direct revenue loss. By predicting which customers are likely to leave, the business can deploy targeted retention campaigns (e.g., discounts, service bundles) to save the relationship before the cancellation occurs.

## 🎯 Objectives
1. Profile the characteristics of customers who churn versus those who stay.
2. Engineer features that accurately capture customer engagement and financial stability.
3. Build a predictive model to classify at-risk customers.
4. Deliver data-backed business recommendations to reduce overall churn rates.

## 📊 Dataset Description
The dataset contains **7,032 records** and **21 features** spanning three main categories:
*   **Demographics:** Gender, Senior Citizen status, Partners, Dependents.
*   **Account Information:** Tenure (months), Contract type, Payment method, Paperless billing, Monthly charges, Total charges.
*   **Services Subscribed:** Phone, Multiple lines, Internet (DSL/Fiber), Online Security, Tech Support, Streaming TV/Movies.

## 🛠️ Tools & Technologies Used
*   **Programming:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Data Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `LogisticRegression`, `StandardScaler`, `LabelEncoder`)
*   **Metrics:** Confusion Matrix, Classification Report, ROC-AUC

## 🔄 Project Workflow

### 1. Data Cleaning
*   Identified and dropped 11 rows with missing `TotalCharges` (negligible portion of the dataset).
*   Removed `customerID` as it provides no predictive value.
*   Standardized text columns and converted non-numerical object data types to strings for proper encoding.
*   Applied Label Encoding for binary features and One-Hot Encoding (`get_dummies`) for multi-class features.

### 2. Exploratory Data Analysis (EDA)
*   Visualized the baseline churn rate (**26.6% Churn** vs **73.4% Retained**).
*   Plotted feature distributions (Tenure, Monthly Charges, Total Charges).
*   Generated a correlation heatmap to identify initial relationships between features and the target variable.

### 3. Feature Engineering
Intentionally engineered post-encoding to isolate specific behavioral signals:
*   `avg_usage_per_month`: Captures historical customer value (`TotalCharges` / `tenure`). A stable metric less affected by impulse subscriptions.
*   `engagement_score`: Captures ecosystem buy-in by summing all active `_Yes` service columns. 
*   `monthly_charge_variance`: Captures billing volatility (`MonthlyCharges` - `avg_usage_per_month`) to proxy potential "payment shock."

### 4. Modeling & Postdictive Analysis
Two models were evaluated: **Logistic Regression** and **Random Forest**. The data was split 80/20, and a `StandardScaler` was applied to normalize numerical features.

| Model | Overall Accuracy | ROC-AUC | Recall (Churn Class) | Precision (Churn Class) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 78.75% | 0.8316 | 52% | 62% |
| **Random Forest** | 75.76% | 0.8292 | **71%** | 53% |

> **💡 Postdictive Analysis & Model Selection:** 
> While Logistic Regression had slightly higher overall accuracy, **Random Forest was selected as the final model.** Why? Because the Random Forest model achieved a **71% Recall** for actual churners (vs. 52% for LR). In the telecom industry, the cost of a false positive (accidentally sending a loyal customer a discount) is vastly lower than the cost of a false negative (failing to identify a customer who is about to leave). We strategically accepted a higher false-alarm rate (lower precision) to capture a significantly larger pool of true churners.

## 🔍 Key Insights
1. **The Contract Trap:** The single largest predictor of churn is the contract type. **88.6% of all churn comes from customers on Month-to-Month contracts**. 
2. **The Tenure Gap:** Customers who churn leave quickly. The average tenure for a churned customer is **~18 months**, compared to **~38 months** for loyal customers.
3. **Ecosystem Buy-in:** Customers lacking supplementary services like Tech Support and Online Security are highly correlated with churn, indicating that basic-tier customers feel less attached to the company.

### Top 3 Feature Importances (Random Forest):
1. `Contract_Month-to-month` (0.121)
2. `TotalCharges` (0.103)
3. `tenure` (0.101)

## 💼 Business Recommendations
Based on the data, I recommend the business prioritize the following actions:

*   **Launch a Contract Conversion Campaign:** Since Month-to-Month customers account for nearly 90% of churn, the marketing team should aggressively target this segment with incentives (e.g., first-year discounts, free streaming upgrades) to convert them to annual or 2-year contracts.
*   **Implement a "First 90 Days" Onboarding Program:** Churn happens early. A dedicated onboarding sequence should be deployed to educate new customers on the value of their services, heavily promoting the use of Tech Support and Online Security to deepen ecosystem integration.

## 📂 Repository Structure
```text
├── data/
│   └── Data-Telco-Customer-Churn 2.csv   # Raw dataset
├── notebooks/
│   └── telco_churn_analysis.ipynb             # Main Jupyter Notebook                          
└── README.md                            
```

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Telco-Customer-Churn-Analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Telco-Customer-Churn-Analysis
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open `notebooks/telco_churn_analysis.ipynb` and run the cells.

## 🔮 Future Improvements
*   **Hyperparameter Tuning:** Utilize `GridSearchCV` or `RandomizedSearchCV` to optimize the Random Forest model's precision without sacrificing recall.
*   **Dashboarding:** Build an interactive Power BI or Tableau dashboard to allow stakeholders to filter churn risk by demographics.
*   **Financial Impact Simulation:** Assign specific dollar values to Customer Acquisition Cost (CAC) and Retention Cost to calculate the exact ROI of the Random Forest model.

## 📬 Author / Contact
** Tessa Saumu **
*   **LinkedIn:** [Linkedin](www.linkedin.com/in/tessa-saumu)
*   **Portfolio:** [Porfolio](tessas-tech-canvas.lovable.app/)
*   **Email:** [Email](theresia.saumu@gmail.com)
