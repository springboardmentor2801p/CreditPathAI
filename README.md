
# Loan Dataset — Initial Data Preprocessing

## Overview
This repository contains the initial preprocessing workflow for a loan dataset.
The goal of this stage is to clean and transform raw data into a structured
format that can be used for Exploratory Data Analysis (EDA) and machine learning.

This is the data preparation phase of a larger Loan Default Prediction project.

---

## Preprocessing Steps

### 1. Data Loading
- Load dataset using Pandas
- Inspect dataset shape

### 2. Duplicate Handling
- Identify duplicate rows
- Remove duplicate records

### 3. Column Cleaning
Remove unnecessary identifier columns when present:
- id
- customer_id
- name

### 4. Missing Value Treatment
- Numeric columns → Median imputation
- Categorical columns → Mode imputation

### 5. Outlier Treatment
Outliers are treated using the IQR method.

Lower Bound = Q1 − 1.5 × IQR  
Upper Bound = Q3 + 1.5 × IQR  

Values outside the range are capped.

### 6. Feature Engineering
The following features are created when columns exist:
- Loan-Income Ratio
- EMI-Income Ratio

### 7. Encoding
- One-Hot Encoding applied to categorical columns
- Boolean values converted to integers

---

## Output
The preprocessing step generates:

Preprocessed_Loan_Dataset.csv

This dataset will be used in later stages:
- Exploratory Data Analysis (EDA)
- Feature selection
- Machine learning model training

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab

---

## How to Run
1. Open the notebook in Google Colab
2. Upload the dataset CSV file
3. Run all preprocessing cells
4. Download the processed dataset

---

## Author
Yamini
