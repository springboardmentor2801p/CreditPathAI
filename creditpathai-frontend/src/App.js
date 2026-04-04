import React, { useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";
import "./App.css";
import { FaUser, FaUniversity } from "react-icons/fa";

function App() {
  const [view, setView] = useState("user");

  const [form, setForm] = useState({
    loanAmount: "",
    interestRate: "",
    annualIncome: "",
    dtiRatio: "",
    existingLoans: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: Number(e.target.value)
    });
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/risk-score", form);
      setResult(res.data);
    } catch {
      alert("Backend not running");
    }
  };

  return (
    <div className="main">
      <h1 className="title">Credit Risk Dashboard</h1>
      <p className="subtitle">
        Enter your financial details to assess credit risk
      </p>

      {/* Toggle */}
      <div className="toggle">
        <button
          className={view === "user" ? "active" : ""}
          onClick={() => setView("user")}
        >
          <FaUser /> User View
        </button>

        <button
          className={view === "bank" ? "active" : ""}
          onClick={() => setView("bank")}
        >
          <FaUniversity /> Bank View
        </button>
      </div>

      {/* FORM */}
      <div className="card">
        <h2>Enter Details</h2>

        <div className="grid">
          <input name="loanAmount" placeholder="Loan Amount" onChange={handleChange} />
          <input name="interestRate" placeholder="Interest Rate (%)" onChange={handleChange} />
          <input name="annualIncome" placeholder="Annual Income" onChange={handleChange} />
          <input name="dtiRatio" placeholder="DTI Ratio" onChange={handleChange} />
          <input name="existingLoans" placeholder="Existing Loans" onChange={handleChange} />
        </div>

        <button className="btn" onClick={handleSubmit}>
          🔍 Check Risk
        </button>
      </div>

      {/* USER VIEW */}
      {view === "user" && result && (
        <div className="card dashboard">

          <h3><FaUser /> User Dashboard</h3>

          <div className="top-cards">
            <div className="mini-card">
              <h4>Risk Score</h4>
              <p className="big">{result.user.risk_score}</p>
              <span>({Math.round(result.user.risk_score * 100)}%)</span>
            </div>

            <div className="mini-card">
              <h4>Risk Level</h4>
              <p className="medium-text">{result.user.risk_level}</p>
            </div>

            <div className="mini-card">
              <h4>Status</h4>
              <p className={`status ${result.user.risk_level.toLowerCase()}`}>
                {result.user.risk_level === "Low"
                  ? "Good"
                  : result.user.risk_level === "Medium"
                  ? "Needs Attention"
                  : "High Risk"}
              </p>
            </div>
          </div>

          <div className="suggestions">
            <h4>Suggestions</h4>
            <ul>
              {result.user.suggestions.map((s, i) => (
                <li key={i}>✔ {s}</li>
              ))}
            </ul>
          </div>

          <div className="gauge">
            <Plot
              data={[
                {
                  type: "indicator",
                  mode: "gauge+number",
                  value: result.user.risk_score * 100,
                  gauge: {
                    axis: { range: [0, 100] },
                    steps: [
                      { range: [0, 30], color: "#22c55e" },
                      { range: [30, 60], color: "#facc15" },
                      { range: [60, 100], color: "#ef4444" }
                    ]
                  }
                }
              ]}
              layout={{ width: 500, height: 300 }}
            />
          </div>

        </div>
      )}

      {/* BANK VIEW */}
      {view === "bank" && result && (
        <div className="card bank-card">
          <h3><FaUniversity /> Bank Dashboard</h3>

          <div className="top-cards">
            <div className="mini-card">
              <h4>Decision</h4>
              <p className="decision">{result.bank.decision}</p>
            </div>

            <div className="mini-card">
              <h4>Risk Score</h4>
              <p className="big">{result.bank.risk_score}</p>
            </div>

            <div className="mini-card">
              <h4>Expected Loss</h4>
              <p className="loss">₹{result.bank.expected_loss}</p>
            </div>
          </div>

          <div className="chart">
            <Plot
              data={[
                {
                  x: ["Risk Score", "Expected Loss"],
                  y: [result.bank.risk_score, result.bank.expected_loss],
                  type: "bar"
                }
              ]}
              layout={{ width: 500, height: 300 }}
            />
          </div>
        </div>
      )}

    </div>
  );
}

export default App;