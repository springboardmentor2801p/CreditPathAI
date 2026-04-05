import React from "react";

function About() {
  return (
    <div style={{
      maxWidth: "900px",
      margin: "auto",
      padding: "40px",
      background: "#e0f2fe",
      borderRadius: "15px",
      boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
    }}>

      <h1 style={{ color: "#1e3a8a" }}>About CreditPath AI</h1>

      <p style={{ marginTop: "15px", color: "#334155" }}>
        CreditPath AI is an intelligent loan risk prediction system designed
        to assist both borrowers and financial institutions in making smarter decisions.
      </p>

      <h3 style={{ marginTop: "20px" }}>🚀 What We Do</h3>
      <ul>
        <li>Predict borrower risk using AI models</li>
        <li>Provide recommendations for users</li>
        <li>Assist banks in loan approval decisions</li>
        <li>Visualize risk through charts and analytics</li>
      </ul>

      <h3 style={{ marginTop: "20px" }}>🎯 Our Goal</h3>
      <p>
        To reduce financial risk and improve decision-making using data-driven insights.
      </p>

      <h3 style={{ marginTop: "20px" }}>💡 Key Features</h3>
      <ul>
        <li>Dual Interface (User & Bank)</li>
        <li>Real-time Predictions</li>
        <li>Interactive Charts</li>
        <li>Clean and Modern UI</li>
      </ul>

    </div>
  );
}

export default About;