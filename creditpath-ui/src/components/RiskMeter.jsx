import React from "react";
import GaugeComponent from "react-gauge-component";

function RiskMeter({ result }) {

  const probability = result.default_probability * 100;

  return (
    <div className="risk-container">

      <h2>Default Risk Probability</h2>

      <div style={{width: "450px", margin: "auto"}}>

        <GaugeComponent
          type="semicircle"
          arc={{
            subArcs: [
              { limit: 25, color: "#22c55e", label: "Low" },
              { limit: 50, color: "#facc15", label: "Medium" },
              { limit: 75, color: "#f97316", label: "High" },
              { limit: 100, color: "#ef4444", label: "Critical" }
            ]
          }}
          pointer={{
            elastic: true,
            animationDelay: 0
          }}
          value={probability}
        />

      </div>

      <h3>{probability.toFixed(2)}%</h3>

      <hr />

      <h2>Expected Loss</h2>
      <h3>₹ {result.expected_loss.toLocaleString()}</h3>

      <hr />

      <h2>Decision Plan</h2>

      <p><b>Priority:</b> {result.decision_plan.priority}</p>
      <p><b>Assigned Team:</b> {result.decision_plan.assigned_team}</p>
      <p><b>Recovery Channel:</b> {result.decision_plan.recovery_channel}</p>
      <p><b>Follow Up:</b> {result.decision_plan.follow_up_frequency}</p>
      <p><b>Legal Action:</b> {result.decision_plan.legal_action ? "YES" : "NO"}</p>

    </div>
  );
}

export default RiskMeter;