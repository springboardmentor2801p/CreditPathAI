# 🚀 CreditPath AI

CreditPath AI is a Loan Risk Prediction and Decision Support System designed for both users and financial institutions.  
It analyzes financial inputs and provides risk predictions, recommendations, and visual insights.

---

## 📌 Features

### 👤 User Side
- Predicts loan risk level (Low / Medium / High)
- Provides personalized recommendations
- Gives actionable tips to improve financial health
- Displays visual charts (pie, bar, approval rate)

### 🏦 Bank Side
- Calculates default probability
- Estimates expected loss
- Suggests loan status (Approved / Conditional / Reject)
- Provides decision strategies

---

## 🛠️ Tech Stack

- **Frontend:** React.js
- **Backend:** FastAPI
- **Visualization:** Plotly.js
- **API Communication:** Axios
- **Development Tools:** VS Code, Google Colab

---

## 📂 Project Structure


CreditPathAI/
│
├── backend/
│ ├── main.py
│ ├── requirements.txt
│
├── frontend/
│ ├── src/
│ ├── package.json
│
├── notebooks/
│ ├── user_model.ipynb
│ ├── bank_model.ipynb
│
├── docs/
│ ├── report.pdf
│ ├── presentation.pptx
│
└── README.md


---

## ⚙️ How to Run the Project

### 🔹 Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

👉 Runs at: http://127.0.0.1:8000

🔹 Frontend (React)
cd frontend
npm install
npm start

👉 Runs at: http://localhost:3000

📊 Outputs
User Risk Prediction with Recommendations
Bank Risk Analysis with Decision Strategy
Interactive Charts and Dashboards

Home Page
User Prediction
Bank Prediction
Charts
🎯 Future Enhancements
Integrate Machine Learning models
Use real-time financial data
Add authentication system
Deploy as web application

👩‍💻 Author
R. Sai Sharshitha Reddy

🙏 Acknowledgement
Developed as part of Infosys Virtual Internship 6.0
Guided by Infosys Springboard Mentor – Shanmuka Priya
