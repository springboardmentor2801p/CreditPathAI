**🚀 CreditPath AI**

CreditPath AI is a Loan Risk Prediction and Decision Support System designed for both borrowers and financial institutions.
It analyzes financial inputs and provides risk predictions, recommendations, and visual insights to support smarter loan decisions.

###**📌 Features**

**###👤 User Side**
- Predicts loan risk level (Low / Medium / High)
- Provides personalized recommendations
- Gives actionable financial tips
- Displays visual insights (charts & comparisons)
- Tracks financial goals and prediction history

**###🏦 Bank / Admin Side**
- Calculates default probability
- Estimates expected loss
- Analyzes borrower financial profile
- Suggests loan decision (Approve / Monitor)
- Provides recovery strategy recommendations

**###🤖 AI Chatbot**
- Interactive assistant for user queries
- Supports risk prediction through chat
- Provides instant responses and guidance

**###🛠️ Tech Stack**
Frontend: React.js
Backend: FastAPI
Machine Learning: LightGBM
Visualization: Plotly.js
API Communication: Axios
Tools: VS Code, Google Colab

**##📂 Project Structure**
CreditPathAI/
│── data/
│   ├── loan_data_sample.csv
│   ├── loan_data_processed.csv
│
│── notebooks/
│   ├── 1_data_processing.ipynb
│   ├── 2_baseline_model.ipynb
│   ├── 3_advanced_model.ipynb
│   ├── 4_user_recommendation.ipynb
│   ├── 5_bank_recommendation.ipynb
│   ├── 6_user_charts.ipynb
│   ├── 7_bank_charts.ipynb
│   ├── 8_fastapi_logic.ipynb
│
│── models/
│   ├── credit_risk_model_lgbm.pkl
│
│── backend/
│   ├── main.py
│   ├── requirements.txt
│
│── frontend/
│   ├── src/
│   ├── package.json
│
│── docs/
│   ├── report.pdf
│   ├── website.pdf
│
│── README.md

▶️ How to Run the Project
🔹 Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Runs at: http://127.0.0.1:8000

🔹 Frontend (React)
cd frontend
npm install
npm start

Runs at: http://localhost:3000

**###📊 Outputs**
- User Risk Prediction with recommendations
- Bank Risk Analysis with decision strategy
- Default probability & expected loss
- Interactive charts and dashboards

**###🔐 Security**
- Login & registration system
- Role-based access (User / Admin)
- PIN-based verification

**###📌 Modules**
- Home Page
- Authentication (Login / Register)
- User Dashboard
- Admin Dashboard
- User Risk Prediction
- Bank Risk Analysis
- Visual Analytics
- AI Chatbot

**###🔮 Future Enhancements**
- Real-time financial data integration
- Advanced fraud detection system
- Improved explainable AI features
- Deployment as full-scale web application

👩‍💻 Author
- R. Sai Sharshitha Reddy

🙏 Acknowledgement
- Developed as part of Infosys Virtual Internship 6.0
- Guided by Infosys Springboard Mentor – Shanmuka Priya
