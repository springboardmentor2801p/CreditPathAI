# CreditPathAI

CreditPathAI is a project that analyzes credit-related data to help understand customer financial behavior and support future credit risk prediction.

---

## Milestone 1 – Project Setup & Baseline Model

Objective:  
Set up the project structure and create a baseline machine learning model.

**Tasks Completed:**
- Created project repository and folder structure.
- Loaded dataset for credit analysis.
- Performed initial preprocessing.
- Built a baseline model for credit prediction.

**Tools Used**:
- Python
- Pandas
- Scikit-learn

---

## Milestone 2 – Data Ingestion & Exploratory Data Analysis

Objective:  
Build a data ingestion pipeline and perform exploratory data analysis (EDA).

**Tasks Completed:**
- Imported multiple CSV datasets.
- Created a data ingestion pipeline using Python.
- Stored data in a SQLite database.
- Performed basic EDA and data cleaning.
- Generated cleaned training dataset for further modeling.

**Tools Used:**
- Python
- Pandas
- SQLite

---

## Project Structure
CreditPathAI/
│
├── Data/
│ ├── cleaned_training_data.csv
│ ├── creditpathai.db
│ ├── data_01.csv
│ ├── data_02.csv
│ └── data_for_training.csv
│
├── src/
│ ├── ingestion.py
│ ├── preprocessing.py
│ └── kpi_analysis.py
## Milestone 3 – Baseline Model Development

In this milestone, a baseline machine learning model was developed to predict borrower default risk.

Steps completed:

* Performed feature engineering on the cleaned dataset.
* Split the data into training and testing sets.
* Implemented a Logistic Regression model using Python and scikit-learn.
* Evaluated the model performance using AUC-ROC.

This baseline model helps understand the initial predictive performance of the system.

## Milestone 4 – Advanced Model Training

In this milestone, advanced machine learning models were implemented to improve prediction accuracy.

Steps completed:

* Implemented XGBoost and LightGBM models.
* Performed hyperparameter tuning to improve model performance.
* Compared the results with the baseline Logistic Regression model.
* Achieved better AUC-ROC scores with advanced models.

These models provide improved predictions for identifying high-risk borrowers and help recommend better recovery strategies.


