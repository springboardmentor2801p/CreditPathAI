# 🚀 CreditPathAI – AI-Powered Credit Risk Analysis System

## 📌 Overview

**CreditPathAI** is an end-to-end **Machine Learning-powered credit risk assessment platform** designed to predict the likelihood of loan default.
It combines **data-driven modeling**, **REST APIs**, and an interactive **frontend interface** to support intelligent financial decision-making.

---

## 🎯 Problem Statement

Traditional credit scoring systems are:

* Static and rule-based
* Limited in capturing complex financial patterns
* Prone to high risk and inefficiency

**CreditPathAI** solves this by leveraging machine learning to provide **dynamic, accurate, and scalable credit risk predictions**.

---

## 💡 Key Features

* 🔍 Predicts loan default risk using ML models
* ⚡ Fast REST API for real-time predictions
* 📊 Data preprocessing and feature engineering pipeline
* 🧠 Multiple ML models (Logistic Regression, etc.)
* 🌐 Frontend interface for user interaction
* 📦 Scalable and modular architecture

---

## 🏗️ Tech Stack

### 🔹 Backend

* Python
* FastAPI / Flask
* Scikit-learn
* Pandas, NumPy

### 🔹 Frontend

* React.js
* Tailwind CSS

### 🔹 Tools & Platforms

* Git & GitHub
* VS Code
* Postman / Swagger

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Model Training (Logistic Regression, etc.)
5. Model Evaluation
6. Deployment via API

---

## 📂 Project Structure

```
CreditPathAI/
│
├── src/          # API and ML model
├── frontend/         # React UI
├── model/            # Saved ML models
├── notebooks/        # EDA and experiments
├── README.md
```

---

## ⚙️ Installation & Setup

### 🔹 Clone the Repository

```bash
git clone https://github.com/your-username/CreditPathAI.git
cd CreditPathAI
```

---

### 🔹 Backend Setup

```bash
cd src
pip install -r requirements.txt
uvicorn main:app --reload
```

---

### 🔹 Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

## 🔗 API Endpoint Example

### POST `/risk-score`

```json
{
  "income": 50000,
  "age": 30,
  "loan_amount": 200000,
  "credit_score": 720
}
```

### Response

```json
{
  "risk_score": 0.23,
  "prediction": "Low Risk"
}
```

---

## 📊 Model Performance

* Algorithm: Logistic Regression
* Evaluation Metrics: Accuracy, Precision, Recall
* Optimized to reduce overfitting and improve generalization

---

## 🚀 Future Enhancements

* Add advanced models (XGBoost, Random Forest)
* Deploy on cloud (AWS / Azure)
* Add authentication system
* Improve UI/UX
* Real-time data integration

---

## 👨‍💻 Author

**Roshan Shaik**

---

## 📜 License

This project is for educational and demonstration purposes.
