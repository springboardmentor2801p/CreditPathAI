import React from "react";
import { useNavigate } from "react-router-dom";

function Home() {
  const navigate = useNavigate();

  return (
    <div style={{ padding: "60px", textAlign: "center" }}>

      <h1 style={{ color: "#1e3a8a" }}>CreditPath AI 🚀</h1>
      <p style={{ color: "#475569", marginBottom: "40px" }}>
        Choose your role to continue
      </p>

      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: "40px",
        flexWrap: "wrap"
      }}>

        {/* USER CARD */}
        <div style={{
          background: "#e0f2fe",
          padding: "30px",
          borderRadius: "15px",
          width: "250px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
        }}>
          <h2>👤 User</h2>
          <p>Check your loan eligibility and risk</p>

          <button
            onClick={() => navigate("/predict/user")}
            style={{
              marginTop: "15px",
              padding: "10px 20px",
              background: "#0284c7",
              color: "white",
              border: "none",
              borderRadius: "8px"
            }}
          >
            User Prediction →
          </button>
        </div>

        {/* BANK CARD */}
        <div style={{
          background: "#fef9c3",
          padding: "30px",
          borderRadius: "15px",
          width: "250px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
        }}>
          <h2>🏦 Bank</h2>
          <p>Analyze borrower risk and decisions</p>

          <button
            onClick={() => navigate("/predict/bank")}
            style={{
              marginTop: "15px",
              padding: "10px 20px",
              background: "#ca8a04",
              color: "white",
              border: "none",
              borderRadius: "8px"
            }}
          >
            Bank Prediction→
          </button>
        </div>

      </div>
    </div>
  );
}

export default Home;