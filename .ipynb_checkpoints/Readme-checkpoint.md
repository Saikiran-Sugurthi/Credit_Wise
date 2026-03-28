# 💳 CreditWise — Loan Approval Prediction

A machine learning project that predicts loan approval decisions based on applicant financial and demographic data. The project covers the full ML pipeline — from raw data cleaning to model comparison and feature engineering.

---

## 📌 Problem Statement

Banks and financial institutions face the challenge of evaluating thousands of loan applications efficiently and fairly. This project builds a classification model to predict whether a loan will be **approved or rejected**, based on factors like credit score, income, DTI ratio, employment status, and more.

---

## 📂 Dataset

**File:** `loan_approval_data.csv`

**Key Features:**

| Feature | Description |
|---|---|
| `Applicant_Income` | Monthly income of the primary applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Credit_Score` | Credit score of the applicant |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Applicant's savings amount |
| `Gender` | Gender of the applicant |
| `Education_Level` | Education level (e.g., UG, PG) |
| `Employment_Status` | Employment type |
| `Marital_Status` | Marital status |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Area type of property |
| `Employer_Category` | Category of employer |
| `Loan_Approved` | **Target variable** — Yes / No |

---

## 🔄 Project Workflow
```
Raw Data
   │
   ▼
Missing Value Imputation (Mean / Mode)
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Feature Encoding (LabelEncoder + OneHotEncoder)
   │
   ▼
Correlation Heatmap Analysis
   │
   ▼
Train-Test Split + StandardScaler
   │
   ▼
Model Training & Evaluation
   │
   ▼
Feature Engineering (Polynomial Features)
   │
   ▼
Final Model Selection
```

---

## 🔍 Exploratory Data Analysis

- **Pie chart** of loan approval distribution (Yes vs No)
- **Bar plots** for Gender and Education Level breakdown
- **Histograms** for Applicant Income and Co-applicant Income
- **Box plots** comparing Applicant Income, Credit Score, DTI Ratio, and Savings against loan approval
- **Credit Score vs Loan Approval** histogram — identified as the most influential feature
- **Correlation Heatmap** (15×8 figure) showing feature relationships

> 💡 Key Insight: Credit Score showed the strongest positive correlation with loan approval.

---

## ⚙️ Feature Engineering

- Dropped `Applicant_ID` (non-predictive identifier)
- Applied `LabelEncoder` on ordinal columns: `Education_Level`, `Loan_Approved`
- Applied `OneHotEncoder` (drop first) on nominal columns: `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Created polynomial features: `DTI_Ratio_sq` and `Credit_score_sq` to capture non-linear relationships

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| K-Nearest Neighbors (K=5) | Distance-based classifier |
| Gaussian Naive Bayes | Probabilistic classifier — best performer |

**Metrics evaluated:** Precision, Accuracy, Recall, F1-Score, Confusion Matrix

> ✅ **Best Model: Naive Bayes** — achieved the highest Precision, making it most suitable for deployment in a banking system where false approvals are costly.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — ML models, preprocessing, evaluation
- **matplotlib** — visualizations
- **seaborn** — statistical plots

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Saikiran-Sugurthi/creditwise.git
cd creditwise
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Add the dataset
Place `loan_approval_data.csv` in the root directory.

### 4. Run the notebook
```bash
jupyter notebook credit_wise.ipynb
```

---

## 📁 Repository Structure
```
creditwise/
│
├── credit_wise.ipynb        # Main Jupyter Notebook
├── loan_approval_data.csv   # Dataset (add manually)
└── README.md
```

---

## 🎯 Results Summary

After feature engineering with polynomial terms for DTI Ratio and Credit Score:

- Naive Bayes consistently outperformed other models on **Precision**
- This makes it the most production-ready model for a loan approval system, minimizing incorrect approvals

---

## 📌 Future Improvements

- [ ] Try ensemble models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Build a simple web UI for real-time predictions (Flask / Streamlit)

---

## 🙋 Author

**Your Name**  
[GitHub](https://github.com/Saikiran-Sugurthi) • [LinkedIn](https://linkedin.com/in/sai-kiran-sugurthi-1a267830b)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).