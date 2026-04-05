import React, { useState } from "react";
import axios from "axios";
import Plot from "react-plotly.js";

function Predict({type}) {

  const [userData, setUserData] = useState({
    loan_type: "",
    income: "",
    credit_score: "",
    loan_amount: "",
    missed_payments: ""
  });

  const [bankData, setBankData] = useState({
    loan_amount: "",
    income: "",
    Credit_Score: "",
    LTV: "",
    dtir1: ""
  });

  const [userResult, setUserResult] = useState(null);
  const [bankResult, setBankResult] = useState(null);

  const [loadingUser, setLoadingUser] = useState(false);
  const [loadingBank, setLoadingBank] = useState(false);

  const handleUserChange = (e) => {
    setUserData({ ...userData, [e.target.name]: e.target.value });
  };

  const handleBankChange = (e) => {
    setBankData({ ...bankData, [e.target.name]: e.target.value });
  };

  const predictUser = async () => {
  setLoadingUser(true);

  const formattedData = {
    ...userData,
    income: Number(userData.income),
    credit_score: Number(userData.credit_score),
    loan_amount: Number(userData.loan_amount),
    missed_payments: Number(userData.missed_payments)
  };

  const res = await axios.post("http://127.0.0.1:8000/user-risk", formattedData);

  setUserResult(res.data);
  setLoadingUser(false);
};

  const predictBank = async () => {
  setLoadingBank(true);

  const formattedData = {
    ...bankData,
    loan_amount: Number(bankData.loan_amount),
    income: Number(bankData.income),
    Credit_Score: Number(bankData.Credit_Score),
    LTV: Number(bankData.LTV),
    dtir1: Number(bankData.dtir1)
  };

  const res = await axios.post("http://127.0.0.1:8000/bank-risk", formattedData);

  setBankResult(res.data);
  setLoadingBank(false);
};

  const approvalRate = userResult
  ? userResult.risk_level === "Low"
    ? 90
    : userResult.risk_level === "Medium"
    ? 60
    : 30
  : 0;

  return (
  <div style={{ maxWidth: "1000px", margin: "auto" }}>

    <h1 style={{ textAlign: "center", marginBottom: "30px" }}>
      Credit Risk Prediction Dashboard
    </h1>

      {/* ================= USER CARD ================= */}
    {type==="user" && (
      <div style={{
        background: "#e0f2fe",
        padding: "25px",
        borderRadius: "15px",
        marginBottom: "30px",
        boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
      }}>
      <h2>User (Borrower)</h2>

      <input placeholder="Loan Type" name="loan_type" onChange={handleUserChange} /><br /><br />
      <input placeholder="Income" name="income" onChange={handleUserChange} /><br /><br />
      <input placeholder="Credit Score" name="credit_score" onChange={handleUserChange} /><br /><br />
      <input placeholder="Loan Amount" name="loan_amount" onChange={handleUserChange} /><br /><br />
      <input placeholder="Missed Payments" name="missed_payments" onChange={handleUserChange} /><br /><br />

      <button onClick={predictUser} style={{
        padding: "10px 20px",
        background: "#0284c7",
        color: "white",
        border: "none",
        borderRadius: "8px",
        cursor: "pointer",
        transition: "all 0.3s ease"
      }}>
        Predict User Risk
      </button>
      
      {loadingUser && <p>🔄 Predicting user risk...</p>}

      {userResult && (
        <div style={{ marginTop: "20px", background: "#e0f2fe", padding: "15px", borderRadius: "10px" }}>
    
    <h3>User Result</h3>

    <div style={{
  display: "flex",
  gap: "20px",
  marginTop: "15px",
  flexWrap: "wrap"
}}>

  <div style={{
    padding: "15px",
    background: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    transition: "0.3s"
  }}
  onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
  onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
  >
    <h4>Risk Level</h4>
    <p>{userResult.risk_level}</p>
  </div>

  <div style={{
    padding: "15px",
    background: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    transition: "0.3s"
  }}
  onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
  onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
  >
    <h4>Approval %</h4>
    <p>{approvalRate}%</p>
  </div>

</div>

    <p><b>Risk Level:</b> {userResult.risk_level}</p>

    <p><b>Recommendation Summary:</b></p>
<ul>
  {userResult.recommendation_summary && (
    <>
    <p><b>Recommendation Summary:</b></p>
    <ul>
      {userResult.recommendation_summary.map((line, index) => (
        <li key={index}>{line}</li>
      ))}
    </ul>
  </>
)}
</ul>

    <p><b>Tips:</b></p>
    <ul>
      {userResult.tips.map((tip, index) => (
        <li key={index}>{tip}</li>
      ))}
    </ul>

    <div style={{
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "20px",
  marginTop: "20px"
}}>
    <div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>
    
    <Plot
      data={[
        {
          values: [Number(userData.credit_score), 900 - Number(userData.credit_score)],
          labels: ["Your Score", "Improvement Area"],
          type: "pie"
        }
      ]}
      layout={{
        title: {
          text: `User Risk Level: ${userResult.risk_level}`,
          font: { size: 18 }
        },
        transition: { duration: 800 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
      }}
      style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
      />
</div>

<div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>
    <Plot
  data={[
    {
      type: "pie",
      values: [approvalRate, 100 - approvalRate],
      labels: ["Approved", "Remaining"],
      hole: 0.6,
      textinfo: "percent",   // ⭐ THIS LINE ADDS %
      textposition: "inside"
    }
  ]}
  layout={{
    title: {
      text: `Loan Approval: ${approvalRate}%`
    },
    transition: { duration: 800 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    showlegend: true,
    //height: 300
  }}
  style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
/>
</div>

<div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>
    <Plot
  data={[
    {
      x: ["Credit Score", "Income"],
      y: [
        Number(userData.credit_score),
        Number(userData.income) / 1000
      ],
      type: "bar"
    }
  ]}
  layout={{
    title: {
      text: "User Financial Overview"
    },
    transition: { duration: 800 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    }}
  style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
/>

</div>
</div>
    {userResult && (
      <div style={{
        marginTop: "20px",
        padding: "15px",
        background: "#dcfce7",
        borderRadius: "10px"
      }}>
    <h3>🤖 AI Agent Recommendation (User)</h3>

    <p>
      Based on your profile, our AI suggests the following actions:
    </p>

    <ul>
      {userResult.tips.map((tip, i) => (
        <li key={i}>{tip}</li>
      ))}
    </ul>

    <p>
      <b>Suggestion:</b>{" "}
      {userResult.risk_level === "Low"
        ? "You can proceed with the loan."
        : userResult.risk_level === "Medium"
        ? "Improve financial stability before applying."
        : "Avoid applying for loan at this stage."}
    </p>
  </div>
)}
  </div>
)}
</div>
    )}

         { /* ================= BANK CARD ================= */}
     {type==="bank" &&(
       <div style={{
         background: "#fef9c3",
         padding: "25px",
         borderRadius: "15px",
         boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
        }}>
      <h2>Bank / Institution</h2>

      <input placeholder="Loan Amount" name="loan_amount" onChange={handleBankChange} /><br /><br />
      <input placeholder="Income" name="income" onChange={handleBankChange} /><br /><br />
      <input placeholder="Credit Score" name="Credit_Score" onChange={handleBankChange} /><br /><br />
      <input placeholder="LTV" name="LTV" onChange={handleBankChange} /><br /><br />
      <input placeholder="DTI" name="dtir1" onChange={handleBankChange} /><br /><br />

      <button onClick={predictBank} style={{
        padding: "10px 20px",
        background: "#ca8a04",
        color: "white",
        border: "none",
        borderRadius: "8px",
        cursor: "pointer",
        transition: "all 0.3s ease"
      }}>
        Predict Bank Risk
      </button>

      {loadingBank && <p>🔄 Predicting bank risk...</p>}
      
      {bankResult && (
        <div style={{ marginTop: "20px", background: "#fef9c3", padding: "15px", borderRadius: "10px" }}>
    
    <h3>Bank Result</h3>

<div style={{
  display: "flex",
  gap: "20px",
  marginTop: "15px",
  flexWrap: "wrap"
}}>

  <div style={{
    padding: "15px",
    background: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    transition: "0.3s"
  }}
  onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
  onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
  >
    <h4>Default Probability</h4>
    <p>{bankResult.default_probability}</p>
  </div>

  <div style={{
    padding: "15px",
    background: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    transition: "0.3s"
  }}
  onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
  onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
  >
    <h4>Expected Loss</h4>
    <p>{bankResult.expected_loss}</p>
  </div>

  <div style={{
    padding: "15px",
    background: "#ffffff",
    borderRadius: "10px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.1)",
    transition: "0.3s"
  }}
  onMouseEnter={e => e.currentTarget.style.transform = "scale(1.05)"}
  onMouseLeave={e => e.currentTarget.style.transform = "scale(1)"}
  >
    <h4>Status</h4>
    <p>{bankResult.loan_status}</p>
  </div>

</div>

    <p><b>Default Probability:</b> {bankResult.default_probability}</p>
    <p><b>Expected Loss:</b> {bankResult.expected_loss}</p>
    <p><b>Loan Status:</b> {bankResult.loan_status}</p>

    <p><b>Decision Plan:</b></p>
    <ul>
      <li>Priority: {bankResult.bank_decision.priority}</li>
      <li>Recovery Channel: {bankResult.bank_decision.recovery_channel}</li>
      <li>Follow Up: {bankResult.bank_decision.follow_up}</li>
    </ul>

    <div style={{
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "20px",
  marginTop: "20px"
}}>
    <div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>

    <Plot
      data={[
        {
          x: ["Default Probability", "Expected Loss"],
          y: [
            Number(bankResult.default_probability),
            Number(bankResult.expected_loss)
          ],
          type: "bar"
        }
      ]}
      layout={{
        title: {
          text: `Bank Risk Analysis (${bankResult.loan_status})`,
          font: { size: 18 }
        },
        xaxis: {
          title: "Risk Metrics"
        },
        yaxis: {
          title: "Values"
        },
        transition: { duration: 800 },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        //width: 500,//
        //height: 400//
      }}
      style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
      />
  </div>

<div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>
    <Plot
  data={[
    {
      values: [
        Number(bankResult.default_probability),
        1 - Number(bankResult.default_probability)
      ],
      labels: ["Risk", "Safe"],
      type: "pie"
    }
  ]}
  layout={{
    title: {
      text: "Risk Distribution (Bank)"
    },
    transition: { duration: 800 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    //width: 400,//
    //height: 400//
  }}
  style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
/>
</div>

<div style={{
  background: "#ffffff",
  padding: "15px",
  borderRadius: "12px",
  marginBottom: "10px",
  boxShadow: "0 4px 10px rgba(0,0,0,0.1)"
}}>
    <Plot
  data={[
    {
      x: ["Probability", "Loss"],
      y: [
        Number(bankResult.default_probability),
        Number(bankResult.expected_loss) / 100000   // scaled for visibility
      ],
      type: "bar"
    }
  ]}
  layout={{
    title: {
      text: "Decision Metrics Comparison"
    },
    transition: { duration: 800 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
  }}
  style={{ width: "100%", height: "100%" }}
useResizeHandler={true}
/>

</div>
</div>

    {bankResult && (
      <div style={{
        marginTop: "20px",
        padding: "15px",
        background: "#fee2e2",
        borderRadius: "10px"
      }}>
    <h3>🤖 AI Agent Recommendation (Bank)</h3>

    <p><b>Priority:</b> {bankResult.bank_decision.priority}</p>

    <p><b>Recommended Action:</b></p>
    <ul>
      <li>Channel: {bankResult.bank_decision.recovery_channel}</li>
      <li>Follow Up: {bankResult.bank_decision.follow_up}</li>
    </ul>

    <p>
      <b>Final Suggestion:</b>{" "}
      {bankResult.loan_status === "Approved"
        ? "Approve loan with minimal monitoring."
        : bankResult.loan_status === "Conditionally Approved"
        ? "Approve with strict monitoring."
        : "High risk – require intervention."}
    </p>
  </div>
)}
  </div>
)}
</div>
    )}

    </div>
  );
}

export default Predict;