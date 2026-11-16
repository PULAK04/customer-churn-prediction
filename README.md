# Customer Churn Prediction and Retention Analysis

A machine learning project focused on predicting customer churn using historical telecom customer data. The project explores customer behavior, compares classification algorithms, and identifies patterns that can help businesses understand why customers leave their services.

## Overview

Customer churn is an important business problem, as losing existing customers directly affects revenue and long-term growth.

This project uses historical customer information, including tenure, subscribed services, contract types, and billing details, to predict the likelihood of customer churn. Multiple machine learning models are trained and evaluated to identify the most effective classification approach.

The analysis also explores customer segments with higher churn rates and uses model predictions to identify potential areas for customer retention strategies.

## Dataset

**Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information such as:

- Customer demographics and tenure
- Internet and additional services
- Contract and payment methods
- Monthly and total charges
- Churn status (Yes/No)

**Target Variable:** `Churn`

- `0` – Customer stayed
- `1` – Customer churned

Download the dataset and place the CSV inside the `data/` directory.

Expected filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas, NumPy | Data processing and numerical operations |
| Matplotlib, Seaborn | Exploratory data analysis and visualization |
| Scikit-learn | Model training and evaluation |
| Joblib | Saving trained models |
| Jupyter Notebook | Development and experimentation |

## Implementation

**1. Data Cleaning**
- Inspected missing values, duplicates, and inconsistent data types.
- Converted `TotalCharges` into a numerical feature.
- Handled blank `TotalCharges` values associated with zero-tenure customers.

**2. Exploratory Data Analysis**
- Examined the distribution of churned and retained customers.
- Analyzed churn patterns across contract types, payment methods, and subscribed services.
- Studied the relationship between customer tenure, monthly charges, and churn.

**3. Data Preprocessing**
- Separated numerical and categorical features.
- Applied median and mode imputation where required.
- Used StandardScaler for numerical features.
- Applied One-Hot Encoding to categorical features.
- Used stratified sampling to preserve class distribution.

**4. Model Training**

Implemented and compared three classification algorithms:

- Logistic Regression
- Random Forest Classifier
- HistGradientBoosting Classifier

Class imbalance was addressed using class weights in supported models.

**5. Model Evaluation**

Models were evaluated using:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

PR-AUC and Recall are particularly useful for evaluating the model's ability to identify customers who may churn.

**6. Threshold Optimization**

Instead of relying entirely on the default 0.5 classification threshold, the decision threshold was optimized using validation data to maximize the F1-score.

The selected threshold was then evaluated on an untouched test set.

**7. Model Interpretation**

- Analyzed Random Forest feature importance.
- Identified customer attributes contributing to model predictions.
- Examined the relationship between customer characteristics and churn risk.

**8. Customer Retention Analysis**

- Grouped customers into low-, medium-, and high-risk categories based on predicted churn probabilities.
- Compared predicted risk and actual churn across customer segments.
- Explored potential retention strategies based on customer behavior and contract patterns.

## Project Structure

```text
customer_churn_prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── customer_churn_model.joblib
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── reports/
│   ├── validation_model_comparison.csv
│   ├── final_test_metrics.csv
│   ├── feature_importance.csv
│   ├── retention_segment_analysis.csv
│   └── high_risk_test_customers.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

Model and report files are generated after executing the notebook.

## Getting Started

**Prerequisites:** Python 3.10 or later.

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git

cd customer-churn-prediction
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

**4. Add the dataset**

Download the CSV from Kaggle and place it inside the `data/` directory using the expected filename.

**5. Run the notebook**

```bash
jupyter lab
```

Open `notebooks/customer_churn_prediction.ipynb` and execute the cells sequentially.

## Results and Outputs

After execution, the notebook generates:

| File | Description |
|---|---|
| `customer_churn_model.joblib` | Trained model and selected threshold |
| `validation_model_comparison.csv` | Comparison of classification models |
| `final_test_metrics.csv` | Final model evaluation metrics |
| `feature_importance.csv` | Important features identified by Random Forest |
| `retention_segment_analysis.csv` | Churn analysis across contract segments |
| `high_risk_test_customers.csv` | Customers classified as high risk in the test set |

Actual model performance metrics are generated during execution and saved for further analysis.

## Key Learnings

- Handling missing values and preprocessing mixed numerical and categorical data.
- Understanding class imbalance in binary classification.
- Comparing linear and ensemble-based machine learning algorithms.
- Evaluating models using precision, recall, F1-score, and PR-AUC.
- Optimizing classification thresholds using validation data.
- Interpreting model predictions through feature importance.
- Translating machine learning predictions into business-oriented insights.

## Limitations

The analysis uses a historical telecom dataset and may not generalize to other businesses or customer populations. Feature importance and customer-segment patterns indicate associations, not causation. Retention strategies should be validated through controlled experiments.

