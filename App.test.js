import React, { useState } from "react";
import { Doughnut, Bar } from "react-chartjs-2";
import "chart.js/auto";
import "./styles.css";

export default function App() {
  const [step, setStep] = useState(1);
  const [data, setData] = useState({
    income: "",
    loan: "",
    expenses: "",
    credit: ""
  });

  const [result, setResult] = useState(null);

  const analyze = () => {
    const inc = Number(data.income);
    const loan = Number(data.loan);
    const exp = Number(data.expenses);
    const credit = Number(data.credit);

    const dti = (loan / inc) * 100;

    let score = 0;

    if (credit >= 750) score += 10;
    else if (credit >= 650) score += 25;
    else score += 40;

    if (dti < 30) score += 10;
    else if (dti < 50) score += 25;
    else score += 40;

    if (exp / inc > 0.5) score += 20;

    score = Math.min(100, score);

    const probability = score;
    const loss = Math.round((probability / 100) * loan * 0.1);

    let decision = "APPROVE";
    let priority = "LOW";

    if (probability > 60) {
      decision = "REJECT";
      priority = "HIGH";
    } else if (probability > 30) {
      decision = "APPROVE_WITH_CONDITIONS";
      priority = "MEDIUM";
    }

    setResult({
      probability,
      loss,
      decision,
      priority,
      dti,
      credit
    });

    setStep(2);
  };

  return (
    <div className="container">
      {step === 1 && (
        <div className="card">
          <h2>Borrower Details</h2>

          <input placeholder="Credit Score" onChange={(e) => setData({ ...data, credit: e.target.value })} />
          <input placeholder="Loan Amount" onChange={(e) => setData({ ...data, loan: e.target.value })} />
          <input placeholder="Income" onChange={(e) => setData({ ...data, income: e.target.value })} />
          <input placeholder="Expenses" onChange={(e) => setData({ ...data, expenses: e.target.value })} />

          <button onClick={analyze}>Get Recommendation →</button>
        </div>
      )}

      {step === 2 && result && (
        <>
          <button className="back" onClick={() => setStep(1)}>⬅ Back</button>

          <div className="grid">

            {/* LEFT PANEL */}
            <div className="card">
              <h3>Borrower Details</h3>
              <p>Credit Score: {result.credit}</p>
              <p>DTI: {result.dti.toFixed(1)}%</p>
            </div>

            {/* CENTER PANEL */}
            <div className="card">
              <span className={`badge ${result.priority}`}>
                {result.priority} PRIORITY
              </span>

              <h3>Default Probability</h3>
              <h1>{result.probability}%</h1>

              <p>Decision: {result.decision}</p>
              <p>Expected Loss: ₹{result.loss}</p>
            </div>

            {/* RIGHT PANEL */}
            <div className="card">
              <h3>Risk Insights</h3>
              <p>Recovery: Early Intervention</p>
              <p>Action: Monitor Customer</p>
            </div>

            {/* GAUGE */}
            <div className="card">
              <h3>Risk Gauge</h3>
              <Doughnut
                data={{
                  labels: ["Risk", "Safe"],
                  datasets: [{
                    data: [result.probability, 100 - result.probability],
                    backgroundColor: ["#ff9800", "#4caf50"]
                  }]
                }}
              />
            </div>

            {/* BAR ANALYSIS */}
            <div className="card">
              <h3>Risk Factor Analysis</h3>
              <Bar
                data={{
                  labels: ["DTI", "Credit"],
                  datasets: [{
                    label: "Values",
                    data: [result.dti, result.credit],
                    backgroundColor: ["orange", "green"]
                  }]
                }}
              />
            </div>

          </div>
        </>
      )}
    </div>
  );
}