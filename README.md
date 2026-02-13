Loan Dataset — Initial Data Preprocessing
Overview

This repository contains the initial data preprocessing workflow for a loan dataset.
The purpose of this stage is to clean and transform raw data into a structured format suitable for exploratory data analysis (EDA) and machine learning model development.

This is the data preparation phase of a larger loan default prediction / credit-risk analysis project.

Preprocessing Tasks Performed
Data Loading

Dataset loaded using Pandas

Initial dataset shape inspected

Duplicate Handling

Duplicate rows identified

Duplicate records removed

Column Cleaning

Removes identifier columns when present:

id

customer_id

name

similar non-analytical fields

Missing Value Treatment
Column Type	Method
Numeric	Median Imputation
Categorical	Mode Imputation
Outlier Treatment

Outliers in numeric columns are treated using the IQR method.

Lower = Q1 − 1.5 × IQR
Upper = Q3 + 1.5 × IQR


Values are capped within the acceptable range.

Feature Engineering

The following features are created when relevant columns exist:

Loan-Income Ratio

EMI-Income Ratio

Encoding

One-Hot Encoding applied to categorical columns

Boolean columns converted to integer format

Output

The preprocessing step generates:

Preprocessed_Loan_Dataset.csv


This dataset will be used in later stages of the project:

Exploratory Data Analysis

Feature selection

Model training

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Google Colab

How to Run

Open notebook in Google Colab

Upload dataset CSV file

Run preprocessing cells

Download processed dataset

Author

Yamini
