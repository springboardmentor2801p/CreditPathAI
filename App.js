import React, { useState } from "react";
import { Doughnut, Bar, Line } from "react-chartjs-2";
import "chart.js/auto";
import "./App.css";

function App() {
  const [page, setPage] = useState("home");
  const [role, setRole] = useState("");

  const [income, setIncome] = useState("");
  const [newLoan, setNewLoan] = useState("");
  const [existingLoans, setExistingLoans] = useState("");
  const [expenses, setExpenses] = useState("");
  const [credit, setCredit] = useState("");
  const [employment, setEmployment] = useState("");
  const [savings, setSavings] = useState("");
  const [age, setAge] = useState("");

  const [result, setResult] = useState(null);

  const analyze = () => {
    const inc = Number(income);
    const nLoan = Number(newLoan);
    const eLoans = Number(existingLoans);
    const exp = Number(expenses);
    const cs = Number(credit);
    const save = Number(savings);
    const a = Number(age);

    if (!inc || !nLoan || !exp || !cs || !employment || !save || !a) {
      alert("Please fill all fields correctly");
      return;
    }

    const totalLoan = nLoan + eLoans;

    // Debt-to-Income Ratio
    const dti = (totalLoan / inc) * 100;
    let dtiScore = dti < 40 ? 0 : dti <= 60 ? 20 : 40;

    // Expense Ratio
    const expRatio = (exp / inc) * 100;
    let expScore = expRatio < 40 ? 0 : expRatio <= 60 ? 20 : 40;

    // Credit Score
    let creditScore = cs >= 750 ? 0 : cs >= 650 ? 20 : 40;

    // Employment
    let empScore = employment === "salaried" ? 0 : 10;

    // Savings Ratio
    const saveRatio = (save / inc) * 100;
    let saveScore = saveRatio > 50 ? 0 : saveRatio >= 20 ? 10 : 20;

    // Age Factor
    let ageScore = a >= 25 && a <= 60 ? 0 : 10;

    const totalRisk = dtiScore + expScore + creditScore + empScore + saveScore + ageScore;
    const level = totalRisk <= 35 ? "Low Risk" : totalRisk <= 65 ? "Medium Risk" : "High Risk";

    // Suggestions
    const suggestions = [
      cs < 700 && `Improve your credit score (current: ${cs})`,
      dti > 60 && `Reduce total debt or loan (DTI: ${dti.toFixed(1)}%)`,
      expRatio > 50 && `Control expenses (Expense Ratio: ${expRatio.toFixed(1)}%)`,
      saveRatio < 20 && `Increase savings (Savings Ratio: ${saveRatio.toFixed(1)}%)`,
      employment !== "salaried" && `Provide stable income proof`,
      ageScore > 0 && `Check age eligibility (Current age: ${a})`
    ].filter(Boolean);

    // AI Explanation
    const explanationText = `
Credit Score: ${cs} → ${
      cs >= 750 ? "Excellent" : cs >= 650 ? "Good" : "Poor"
    }
Debt-to-Income: ${dti.toFixed(1)}% → ${
      dti < 40 ? "Low" : dti <= 60 ? "Medium" : "High"
    }
Expense Ratio: ${expRatio.toFixed(1)}% → ${
      expRatio < 40 ? "Low" : expRatio <= 60 ? "Medium" : "High"
    }
Employment: ${employment} → ${employment === "salaried" ? "Stable" : "Risky"}
Savings Ratio: ${saveRatio.toFixed(1)}% → ${
      saveRatio > 50 ? "High" : saveRatio >= 20 ? "Medium" : "Low"
    }
Age: ${a} → ${ageScore === 0 ? "Eligible" : "Check limits"}
Overall Risk Level: ${level}
    `;

    setResult({
      totalRisk,
      level,
      dti, dtiScore,
      expRatio, expScore,
      creditScore,
      empScore,
      saveRatio, saveScore,
      ageScore,
      suggestions,
      explanationText
    });

    setPage("result");
  };

  return (
    <div className="app">

      {/* HOME PAGE */}
      {page === "home" && (
        <div className="center">
          <div className="card home-card">
            <h1>💳 CreditPathAI Loan Dashboard</h1>
            <p>Interactive Risk Assessment & Explainable AI</p>
            <div className="home-buttons">
              <button className="btn blue" onClick={() => { setRole("bank"); setPage("form"); }}>🏦 Bank</button>
              <button className="btn yellow" onClick={() => { setRole("user"); setPage("form"); }}>👤 Applicant</button>
            </div>
          </div>
        </div>
      )}

      {/* FORM PAGE */}
      {page === "form" && (
        <div className="center">
          <div className="card form-card">
            <h2 className="form-title">Enter Financial Details</h2>

            <div className="form-group">
              <label>Income</label>
              <input type="number" className="input" placeholder="Total Monthly/Annual Income" onChange={(e) => setIncome(e.target.value)} />
              <small>Enter your gross income</small>
            </div>

            <div className="form-group">
              <label>New Loan Amount</label>
              <input type="number" className="input" placeholder="Loan you want to take" onChange={(e) => setNewLoan(e.target.value)} />
            </div>

            <div className="form-group">
              <label>Existing Loans</label>
              <input type="number" className="input" placeholder="Any outstanding loans" onChange={(e) => setExistingLoans(e.target.value)} />
            </div>

            <div className="form-group">
              <label>Monthly/Annual Expenses</label>
              <input type="number" className="input" placeholder="All regular expenses" onChange={(e) => setExpenses(e.target.value)} />
            </div>

            <div className="form-group">
              <label>Credit Score</label>
              <input type="number" className="input" placeholder="650 - 900" onChange={(e) => setCredit(e.target.value)} />
              <small>Higher score improves loan approval</small>
            </div>

            <div className="form-group">
              <label>Employment Status</label>
              <select className="input" onChange={(e) => setEmployment(e.target.value)}>
                <option value="">Select Employment</option>
                <option value="salaried">Salaried</option>
                <option value="self">Self-employed</option>
              </select>
            </div>

            <div className="form-group">
              <label>Savings</label>
              <input type="number" className="input" placeholder="Current savings" onChange={(e) => setSavings(e.target.value)} />
            </div>

            <div className="form-group">
              <label>Age</label>
              <input type="number" className="input" placeholder="Your age" onChange={(e) => setAge(e.target.value)} />
            </div>

            <div className="form-buttons">
              <button className="btn blue" onClick={analyze}>Analyze Risk</button>
              <button className="btn" onClick={() => setPage("home")}>⬅ Back</button>
            </div>
          </div>
        </div>
      )}

      {/* RESULT PAGE */}
      {page === "result" && result && (
        <div className="dashboard">
          <div className="box">
            <h4>Risk Level</h4>
            <p className={result.totalRisk <= 35 ? "low" : result.totalRisk <= 65 ? "medium" : "high"}>{result.level}</p>
          </div>

          <div className="box">
            <h4>Risk Score</h4>
            <h3>{result.totalRisk}%</h3>
          </div>

          <div className="box chart">
            <Doughnut data={{
              datasets: [{ data: [result.totalRisk, 100 - result.totalRisk], backgroundColor: ["#ef4444", "#1e293b"] }]
            }} />
          </div>

          <div className="box chart">
            <h4>Income vs Loan vs Expenses</h4>
            <Bar data={{
              labels: ["Income", "New Loan", "Existing Loans", "Expenses", "Savings"],
              datasets: [{
                data: [income, newLoan, existingLoans, expenses, savings],
                backgroundColor: ["#3b82f6", "#f97316", "#eab308", "#22c55e", "#8b5cf6"]
              }]
            }} />
          </div>

          <div className="box chart">
            <h4>Risk Factors</h4>
            <Bar data={{
              labels: ["Credit", "DTI", "Expenses", "Employment", "Savings", "Age"],
              datasets: [{
                data: [result.creditScore, result.dtiScore, result.expScore, result.empScore, result.saveScore, result.ageScore],
                backgroundColor: ["#3b82f6", "#f97316", "#22c55e", "#eab308", "#8b5cf6", "#ef4444"]
              }]
            }} />
          </div>

          <div className="box chart">
            <h4>Risk Trend</h4>
            <Line data={{
              labels: ["Credit", "DTI", "Expenses", "Employment", "Savings", "Age"],
              datasets: [{
                label: "Risk Contribution",
                data: [result.creditScore, result.dtiScore, result.expScore, result.empScore, result.saveScore, result.ageScore],
                fill: false,
                borderColor: "#f97316",
                tension: 0.3
              }]
            }} />
          </div>

          <div className="box">
            <h4>AI Analysis</h4>
            <pre style={{ whiteSpace: "pre-line", fontSize: "13px" }}>{result.explanationText}</pre>
          </div>

          <div className="box">
            <h4>Suggestions</h4>
            <ul>
              {result.suggestions.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>

          <div className="box">
            <h4>Decision</h4>
            <p>{result.totalRisk <= 35 ? "Approve ✅" : result.totalRisk <= 65 ? "Approve with Conditions ⚠️" : "Reject ❌"}</p>
          </div>

          <div className="box">
            <button className="btn blue" onClick={() => setPage("home")}>⬅ Back</button>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;