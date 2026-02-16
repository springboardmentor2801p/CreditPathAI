<div align="center">

# 🚀 CreditPathAI – Smart Loan Recovery System

![Typing Animation](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=2E8B57&center=true&vCenter=true&width=700&lines=AI-Powered+Loan+Default+Prediction;Smart+Risk+Assessment+%26+Recovery;Infosys+Internship+Project;ML-Driven+Financial+Solutions;by+ronit+shaw)

<br>

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge&logo=github)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-red?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

</div>

---

## 📋 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🎯 Business Objective](#-business-objective)
- [📊 Dataset Information](#-dataset-information)
- [⚙️ Project Workflow](#️-project-workflow)
- [📈 Key Performance Indicators](#-key-performance-indicators)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📊 Data Processing Summary](#-data-processing-summary)
- [✨ Features & Capabilities](#-features--capabilities)
- [🔮 Next Steps & Roadmap](#-next-steps--roadmap)
- [👨‍💻 Author & Acknowledgments](#-author--acknowledgments)

---

## 📌 Project Overview

CreditPathAI is an intelligent machine learning system designed to predict borrower default risk and enable smarter loan recovery decisions. By leveraging advanced data preprocessing, feature engineering, and predictive analytics, this system empowers financial institutions to identify high-risk borrowers early and optimize their recovery strategies.

### Key Highlights:

✅ **Data-Driven Approach** – Comprehensive data ingestion and preprocessing pipeline  
✅ **Feature Optimization** – 13 engineered features from raw dataset  
✅ **Scalability** – Handles 148K+ loan records efficiently  
✅ **Business Alignment** – Metrics tied directly to business KPIs  
✅ **Ready for Deployment** – Modular architecture for production use

---

## 🎯 Business Objective

### Primary Goals:

1. **Early Risk Identification** – Detect high-risk borrowers before default occurs
2. **Loss Mitigation** – Reduce default exposure through proactive recovery prioritization
3. **Resource Optimization** – Allocate recovery resources to highest-impact targets
4. **Decision Support** – Provide data-driven insights for lending decisions

### Expected Impact:

- 🎯 Reduce default rate by identifying risky profiles early
- 💰 Minimize financial losses through targeted interventions
- 📊 Improve recovery efficiency with predictive scoring
- 🤖 Enable automated risk-based decision making

---

## 📊 Dataset Information

### Raw Dataset Characteristics:

| Metric | Value |
|--------|-------|
| **Source** | Kaggle Loan Default Dataset |
| **Initial Records** | 148,671 |
| **Initial Features** | 34 |
| **Data Format** | CSV |
| **Target Variable** | Loan Default Status (Binary) |

### Processed Dataset Characteristics:

| Metric | Value |
|--------|-------|
| **Final Records** | 98,188 ✅ |
| **Total Features** | 47 ✅ |
| **Missing Data** | 0% (Cleaned) |
| **Duplicates Removed** | Yes ✅ |
| **Categorical to Numerical** | 100% Converted |

### Data Quality Metrics:

```
📈 Data Transformation Summary:
├── Records Retained: 98,188 / 148,671 (66.1%)
├── Feature Expansion: 34 → 47 (+38.2%)
├── Null Values Removed: 50,483 rows
├── Duplicate Rows Removed: Zero tolerance applied
├── Categorical Features Encoded: 18/18 (100%)
└── Numerical Features Normalized: 29/29 (100%)
```

---

## ⚙️ Project Workflow

### 1️⃣ Data Ingestion Phase
```
📥 Data Loading
├── Load CSV into Pandas DataFrame
├── Verify structure & data types
├── Initial exploratory inspection
├── Memory optimization
└── Data validation checks
```

**Completed:** ✅

### 2️⃣ Data Preprocessing Phase
```
🧹 Cleaning & Transformation
├── Missing Value Treatment
│   ├── Identify missing data patterns
│   ├── Apply domain-specific imputation
│   └── Document removal rationale
├── Duplicate Removal
│   ├── Identify duplicate records
│   └── Remove with zero-tolerance approach
├── Categorical Encoding
│   ├── One-Hot Encoding for nominal features
│   ├── Ordinal Encoding for ordinal features
│   └── Target Encoding for high-cardinality features
├── Feature Scaling
│   ├── StandardScaler for numerical features
│   ├── RobustScaler for outlier-prone features
│   └── MinMaxScaler for bounded features
└── Dataset Export
    └── Save cleaned CSV for modeling
```

**Completed:** ✅

### 3️⃣ Feature Engineering Phase
```
🔧 Feature Enhancement
├── Domain-Specific Features
│   ├── Credit Utilization Ratio
│   ├── Payment-to-Income Ratio
│   ├── Debt-to-Income Ratio
│   └── Savings to Loan Ratio
├── Behavioral Indicators
│   ├── Payment Frequency Score
│   ├── Delinquency History
│   ├── Account Age Metrics
│   └── Account Activity Trend
├── Risk Indicators
│   ├── Credit Risk Score
│   ├── Loan Amount Percentile
│   └── Monthly Installment Ratio
└── Temporal Features
    ├── Loan Duration Estimation
    └── Age-Based Segments
```

**Status:** In Progress 🔄

### 4️⃣ Exploratory Data Analysis (EDA)
```
📊 Analysis Components
├── Univariate Analysis
│   ├── Numerical distributions (histograms, box plots)
│   ├── Categorical distributions (bar charts)
│   └── Summary statistics
├── Bivariate Analysis
│   ├── Feature correlations
│   ├── Default rate by feature
│   └── Cross-tabulation analysis
├── Multivariate Analysis
│   ├── Feature interactions
│   ├── Cluster analysis
│   └── Principal Component Analysis
└── Business Insights
    ├── Risk segment identification
    └── Pattern recognition
```

**Status:** Completed ✅

### 5️⃣ Model Development Phase (Planned)
```
🤖 Modeling Pipeline
├── Model Selection
│   ├── Logistic Regression (baseline)
│   ├── Random Forest
│   ├── XGBoost
│   └── LightGBM
├── Hyperparameter Tuning
│   ├── Grid Search / Random Search
│   ├── Cross-Validation (5-Fold)
│   └── Performance Optimization
├── Model Evaluation
│   ├── AUC-ROC Curve Analysis
│   ├── Confusion Matrix Analysis
│   ├── Feature Importance Ranking
│   └── Business Impact Assessment
└── Model Selection & Finalization
    └── Choose best performer
```

**Status:** Upcoming 📅

---

## 📈 Key Performance Indicators

### 🔹 Model Performance KPIs

| KPI | Target | Importance |
|-----|--------|-----------|
| **AUC-ROC Score** | > 0.85 | High |
| **Accuracy** | > 80% | High |
| **Precision** | > 75% | Critical |
| **Recall** | > 70% | Critical |
| **F1-Score** | > 0.72 | High |
| **Specificity** | > 85% | Medium |

### 🔹 Business KPIs

| KPI | Metric | Status |
|-----|--------|--------|
| **Default Rate** | % of defaults in dataset | 📊 Analyzed |
| **Average Loan Amount** | Mean loan disbursed | 📊 Analyzed |
| **Credit Score Distribution** | Borrower credit profiles | 📊 Analyzed |
| **High-Risk Borrower %** | Percentage of risky profiles | 📊 Analyzed |
| **Loan Status Distribution** | Active/Closed/Default breakdown | 📊 Analyzed |
| **Recovery Rate** | Successfully recovered defaults | 🔄 Planned |
| **Cost-to-Benefit Ratio** | Recovery investment ROI | 🔄 Planned |

### 📉 Expected Outcomes:

- **Precision Focus:** Minimize false positives (non-defaulters marked as risky)
- **Recall Priority:** Capture maximum true defaults
- **Business Impact:** Improve recovery efficiency by 30-40%
- **Scalability:** Process new applications in real-time

---

## 🛠️ Tech Stack

<div align="center">

### Languages & Frameworks

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-f37726?style=flat-square&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/Git-Version%20Control-f1502f?style=flat-square&logo=git&logoColor=white)

### Data Processing & ML Libraries

![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-00a0d2?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient-9b59b6?style=flat-square)

### Data Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-0173b2?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical-76b900?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3f4f75?style=flat-square&logo=plotly)

### Development Environment

![VS Code](https://img.shields.io/badge/VS%20Code-Editor-007acc?style=flat-square&logo=visual-studio-code)
![Anaconda](https://img.shields.io/badge/Anaconda-Environment-44a833?style=flat-square&logo=anaconda)

</div>

---

## 📁 Project Structure

```
CreditPathAI/
│
├── 📂 data/
│   ├── Loan_Default.csv                    # Original dataset (148.6K records)
│   ├── Loan_Default_cleaned.csv            # Processed dataset (98.1K records)
│   ├── data_dictionary.md                  # Feature descriptions
│   └── preprocessing_log.txt               # Cleaning operations log
│
├── 📂 notebooks/
│   ├── 01_data_ingestion.ipynb             # Data loading & validation
│   ├── 02_data_preprocessing.ipynb         # Cleaning & transformation
│   ├── 03_exploratory_data_analysis.ipynb  # EDA & visualizations
│   ├── 04_feature_engineering.ipynb        # Feature creation
│   └── 05_model_development.ipynb          # Model training (Upcoming)
│
├── 📂 src/
│   ├── preprocessing.py                    # Data cleaning functions
│   ├── feature_engineering.py              # Feature creation functions
│   ├── utils.py                            # Utility functions
│   └── config.py                           # Configuration settings
│
├── 📂 models/
│   ├── best_model.pkl                      # Trained model (Upcoming)
│   ├── model_performance.json              # Model metrics
│   └── feature_importance.csv              # Feature rankings
│
├── 📂 reports/
│   ├── eda_report.html                     # Interactive EDA report
│   ├── data_quality_report.txt             # Data quality summary
│   └── business_insights.md                # Key findings
│
├── 📂 dashboards/
│   ├── dashboard.py                        # Streamlit dashboard (Upcoming)
│   └── api.py                              # Flask API (Upcoming)
│
├── 📄 README.md                            # Project documentation
├── 📄 requirements.txt                     # Python dependencies
├── 📄 setup.py                             # Package setup
├── 📄 .gitignore                           # Git ignore file
└── 📄 LICENSE                              # MIT License

```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git (optional)

### Installation Steps

**Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/CreditPathAI.git
cd CreditPathAI
```

**Step 2: Create Virtual Environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n creditpathai python=3.8
conda activate creditpathai
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Launch Jupyter Notebook**
```bash
jupyter notebook
```

**Step 5: Run Notebooks in Order**
```
1. 01_data_ingestion.ipynb
2. 02_data_preprocessing.ipynb
3. 03_exploratory_data_analysis.ipynb
4. 04_feature_engineering.ipynb
```

### Sample Requirements.txt

```
pandas==1.3.5
numpy==1.21.6
scikit-learn==0.24.2
xgboost==1.5.2
lightgbm==3.3.2
matplotlib==3.5.1
seaborn==0.11.2
plotly==5.5.0
jupyter==1.0.0
jupyter-notebook==6.4.10
ipython==7.30.1
python-dotenv==0.19.0
```

---

## 📊 Data Processing Summary

### Data Quality Improvements

```
🔄 Preprocessing Pipeline Summary:

INPUT DATA
│
├─ 📥 Data Ingestion
│  └─ 148,671 records × 34 features
│
├─ 🧹 Missing Value Treatment
│  ├─ Identified: 50,483 rows with nulls
│  ├─ Strategy: Domain-specific imputation + removal
│  └─ Result: 98,188 records retained (66.1%)
│
├─ 🔄 Duplicate Removal
│  ├─ Found: 0 exact duplicates
│  ├─ Threshold: Zero tolerance
│  └─ Result: All unique records
│
├─ 🏷️ Categorical Encoding
│  ├─ One-Hot: 12 features
│  ├─ Ordinal: 4 features
│  ├─ Label: 2 features
│  └─ Result: 100% numerical dataset
│
├─ 📏 Feature Scaling
│  ├─ StandardScaler: 15 features
│  ├─ RobustScaler: 8 features
│  ├─ MinMaxScaler: 6 features
│  └─ Result: Normalized feature space
│
└─ 💾 Export
   └─ Loan_Default_cleaned.csv (98,188 × 47)

OUTPUT DATA
```

### Feature Summary

| Category | Count | Status |
|----------|-------|--------|
| Numerical Features | 29 | ✅ Normalized |
| Categorical Features | 18 | ✅ Encoded |
| Engineered Features | 13 | 🔄 In Progress |
| **Total Features** | **47** | ✅ Complete |

---

## ✨ Features & Capabilities

### 🎯 Current Capabilities

✅ **Robust Data Pipeline**
- Automated data ingestion from multiple formats
- Intelligent missing value handling
- Duplicate detection and removal
- Data validation and quality checks

✅ **Advanced Preprocessing**
- Categorical feature encoding (One-Hot, Ordinal, Label)
- Numerical feature scaling and normalization
- Outlier detection and treatment
- Feature interaction identification

✅ **Feature Engineering**
- Financial ratio calculations (debt-to-income, utilization)
- Behavioral scoring metrics
- Risk indicators
- Temporal features

✅ **Exploratory Analysis**
- Univariate, bivariate, and multivariate analysis
- Correlation analysis and heatmaps
- Distribution analysis with statistical tests
- Default rate analysis by segments

### 🚀 Upcoming Features

🔄 **Predictive Modeling**
- Multiple model implementations (LR, RF, XGBoost, LightGBM)
- Hyperparameter optimization
- Cross-validation and evaluation
- Feature importance analysis

🔄 **Deployment Components**
- REST API for real-time predictions
- Interactive Streamlit dashboard
- Model serving with Docker
- CI/CD pipeline integration

🔄 **Advanced Analytics**
- SHAP explainability
- Feature interaction analysis
- Cluster-based risk segmentation
- Causal inference modeling

---

## 🔮 Next Steps & Roadmap

### Phase 2: Model Development (Feb - Mar 2024)
- [ ] Complete feature engineering
- [ ] Train baseline models
- [ ] Perform hyperparameter tuning
- [ ] Evaluate and compare models
- [ ] Select best performing model
- [ ] Document model specifications

### Phase 3: Deployment (Apr - May 2024)
- [ ] Create Flask/FastAPI REST API
- [ ] Build Streamlit dashboard
- [ ] Containerize with Docker
- [ ] Set up CI/CD pipeline
- [ ] Performance monitoring setup

### Phase 4: Optimization & Scaling (Jun 2024)
- [ ] Model performance optimization
- [ ] Database integration
- [ ] Real-time prediction capability
- [ ] User interface enhancement
- [ ] Production deployment

### Phase 5: Advanced Features (Future)
- [ ] SHAP/LIME model interpretability
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] Advanced visualization suite
- [ ] Multi-model ensemble approach

---

## 📚 Key Insights & Findings

### Data Insights

📊 **Default Distribution**
- Overall default rate: [To be analyzed]
- Default concentration by income bracket: [To be analyzed]
- Geographic default patterns: [To be analyzed]

💰 **Loan Characteristics**
- Average loan amount: [To be analyzed]
- Loan amount vs. default correlation: [To be analyzed]
- Optimal lending range: [To be analyzed]

👥 **Borrower Profiles**
- High-risk demographics: [To be analyzed]
- Credit score distribution: [To be analyzed]
- Employment stability impact: [To be analyzed]

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/CreditPathAI.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit changes**
   ```bash
   git commit -m "Add descriptive message"
   ```

4. **Push to branch**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request**

---

## 📖 Documentation

### Additional Resources

- 📘 [Data Dictionary](./data/data_dictionary.md) - Feature descriptions
- 📊 [EDA Report](./reports/eda_report.html) - Interactive visualizations
- 🔍 [Preprocessing Log](./data/preprocessing_log.txt) - Detailed cleaning steps
- 💡 [Business Insights](./reports/business_insights.md) - Key findings

---

## 📞 Support & Contact

### Need Help?

- 📧 **Email:** your.email@infosys.com
- 💼 **LinkedIn:** [Your LinkedIn Profile](https://linkedin.com)
- 🐙 **GitHub:** [Your GitHub Profile](https://github.com)
- 📱 **Phone:** [Your Contact Number]

### Report Issues

Found a bug? Have a suggestion? Please open an issue on GitHub:
[Create an Issue](https://github.com/yourusername/CreditPathAI/issues)

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author & Acknowledgments

### 👤 Project Author

**Your Name**
- 🎓 **Role:** AI/ML Intern
- 🏢 **Organization:** Infosys Limited
- 📧 **Email:** your.email@infosys.com
- 🔗 **LinkedIn:** [LinkedIn Profile](https://linkedin.com)
- 🐙 **GitHub:** [GitHub Profile](https://github.com)

### 🙏 Acknowledgments

- **Infosys Internship Program** for the opportunity and mentorship
- **Kaggle** for the comprehensive loan default dataset
- **Open Source Community** for excellent ML libraries
- **My Mentors** for guidance and support

### Special Thanks

- 👨‍🏫 Project Mentor: [Mentor Name]
- 👥 Team Members: [Team Names]
- 🤝 Collaborators: [Collaborator Names]

---

<div align="center">

## 🌟 Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

![Star Badge](https://img.shields.io/github/stars/yourusername/CreditPathAI?style=social)
![Fork Badge](https://img.shields.io/github/forks/yourusername/CreditPathAI?style=social)

---

### 📊 Project Statistics

![Lines of Code](https://img.shields.io/badge/Lines%20of%20Code-5K+-blue)
![Files](https://img.shields.io/badge/Project%20Files-15+-green)
![Documentation](https://img.shields.io/badge/Documentation-100%25-brightgreen)
![Datasets](https://img.shields.io/badge/Datasets-1-orange)

---

**Last Updated:** February 2024  
**Status:** 🔄 Active Development

Made with ❤️ by Your Name

</div>
