
---

# CreditPathAI

CreditPathAI is a machine learning project designed to predict loan default risk using borrower financial and demographic information. The system analyzes historical loan data and builds predictive models to identify borrowers who are likely to default.

The goal of this project is to support financial institutions in credit risk assessment by providing a data-driven approach to loan approval and risk management.

This project was developed as part of the Infosys Springboard Internship.

---

# Problem Definition

Financial institutions suffer significant losses due to borrower loan defaults. Early identification of high-risk borrowers is essential to reduce financial risk and improve recovery efficiency.

This project builds a machine learning pipeline capable of predicting whether a borrower will default based on financial and demographic attributes.

The system analyzes historical loan data and learns patterns that distinguish defaulters from non-defaulters.

---

# Project Pipeline

The project follows a complete machine learning workflow including data ingestion, preprocessing, model development, and evaluation.

The major stages include:

1. Data ingestion
2. Data validation
3. Exploratory data analysis
4. Feature engineering
5. Data preprocessing
6. Model training
7. Model evaluation
8. Hyperparameter tuning
9. Risk prediction and recommendation engine

---

# Data Ingestion and Initial Analysis

The dataset is first loaded and analyzed to understand its structure.

The dataset contains borrower related financial information that includes both numerical and categorical variables. Initial exploration focuses on understanding dataset dimensions, column structure, and data types.

Key checks performed include:

Dataset dimensions
Column structure
Data types
Duplicate records
Target variable distribution

The dataset contains a sufficient number of records and multiple features suitable for machine learning analysis.

---

# Data Storage Using SQLite

To improve data management and querying efficiency, the dataset is stored in a SQLite database.

The workflow includes:

CSV dataset ingestion
Conversion to SQLite tables
Exploratory analysis using SQL queries

This step allows structured access to the dataset and demonstrates integration of database management with machine learning pipelines.

---

# Data Validation

Data validation ensures the integrity of the dataset before model training.

The validation process includes:

Checking for duplicate records
Verifying dataset structure
Ensuring correct data types for each feature

Removing duplicates and validating data consistency helps prevent bias and improves model reliability.

---

# Exploratory Data Analysis

Exploratory Data Analysis is performed to understand relationships between borrower attributes and loan default behavior.

Several financial indicators are analyzed.

Credit Score vs Loan Default
Borrowers with lower credit scores show higher default risk.

Income vs Loan Default
Lower income levels are associated with higher probability of loan default.

Loan Amount vs Loan Default
Higher loan amounts may increase financial burden and potential default risk.

Loan Type vs Loan Default
Different loan categories show varying levels of risk.

These insights help identify the most influential factors affecting loan repayment behavior.

---

# Feature Engineering

Machine learning models require numerical input features. Feature engineering transforms categorical attributes into numerical form.

Two encoding methods are used.

Label Encoding
Binary categorical variables are converted into numeric values.

One Hot Encoding
Categorical variables with multiple categories are converted into dummy variables.

These transformations allow machine learning algorithms to process categorical information correctly.

---

# Dataset Preparation

After preprocessing, the dataset is separated into features and target variable.

Target variable: Loan Default Status

Feature matrix contains borrower financial and demographic attributes.

The dataset is then divided into training and testing sets.

Training Data
80 percent of the dataset

Testing Data
20 percent of the dataset

Stratified sampling is applied to maintain the original class distribution.

Training dataset size
118,936 records with 46 features

Testing dataset size
29,734 records with 46 features

---

# Feature Scaling

Standardization is applied using StandardScaler.

Standardization centers the data around zero mean and unit variance.

This improves model convergence and ensures that all features contribute proportionally during model optimization.

Feature scaling is particularly important for models such as Logistic Regression.

---

# Machine Learning Models

Multiple machine learning models are trained and evaluated to predict loan default risk.

Logistic Regression
XGBoost
LightGBM

Each model is evaluated using classification performance metrics.

---

# Logistic Regression Baseline Model

Logistic Regression is used as the baseline model because it is simple, interpretable, and effective for binary classification.

The model uses class balancing to address potential class imbalance.

Performance metrics:

ROC AUC Score: 0.825
Precision: 0.559
Recall: 0.685
F1 Score: 0.615

The model provides moderate predictive performance and serves as a benchmark for comparison with more advanced models.

---

# Confusion Matrix Analysis

The confusion matrix reveals the classification behavior of the baseline model.

True Negatives: 18,442
True Positives: 5,019
False Negatives: 2,309
False Positives: 3,964

Although the model captures a significant portion of defaulters, some high risk borrowers are still missed, indicating the need for stronger models.

---

# XGBoost Model

XGBoost is used as an advanced ensemble model capable of capturing complex nonlinear relationships.

Performance metrics:

Accuracy: 93.28 percent
Precision: 0.959
Recall: 0.759
F1 Score: 0.848
ROC AUC Score: 0.972

The results show significant improvement over the baseline logistic regression model.

---

# Hyperparameter Tuning

Hyperparameter tuning is performed using RandomizedSearchCV.

The goal of tuning is to optimize model parameters, improve generalization, and maximize predictive performance.

After tuning, the XGBoost model achieved:

ROC AUC Score: 0.9766

This improvement confirms the importance of parameter optimization.

---

# LightGBM Model

LightGBM is evaluated as another gradient boosting model optimized for efficiency and high performance.

Performance metrics:

Accuracy: 94.59 percent
Precision: 0.948
Recall: 0.826
F1 Score: 0.883
ROC AUC Score: 0.986

Among all models tested, LightGBM provides the best predictive performance and strongest class separation capability.

---

# Overfitting Check

Model generalization is verified by comparing training and testing ROC AUC scores.

A small difference between training and testing performance indicates minimal overfitting and strong generalization ability.

---

# Risk Prediction and Recommendation Engine

The trained model is used to estimate loan default probability for individual borrowers.

Based on predicted risk levels, the system can provide recommendations for credit risk management.

For example:

A borrower with very low predicted default probability was classified as low risk.

Despite a relatively high loan amount, the expected financial loss was extremely small. Therefore, the system recommended a simple automated reminder rather than aggressive recovery actions.

This demonstrates how machine learning predictions can be translated into actionable financial decision support.

---

# Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit Learn
XGBoost
LightGBM
SQLite
Jupyter Notebook

---

# Project Structure

CreditPathAI

Creditpathloan.ipynb
Loan_Default.csv
Loan_Default_Preprocessed.csv
README.md
.gitignore

---

# Author

Joe Sharwin
MSc Artificial Intelligence
Infosys Springboard Internship Project

---


Those two upgrades make the project look **10x more professional on GitHub**.
