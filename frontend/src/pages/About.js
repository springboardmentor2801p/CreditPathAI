import React from "react";

const techStack = [
  { icon: "⚛️", label: "React.js" },
  { icon: "🐍", label: "FastAPI" },
  { icon: "🤖", label: "LightGBM" },
  { icon: "📊", label: "Plotly.js" },
  { icon: "🔗", label: "Axios" },
  { icon: "🧪", label: "Pandas" },
  { icon: "💾", label: "Joblib" },
  { icon: "🛣️", label: "React Router" },
];

function About() {
  return (
    <div className="about-page">

      {/* Hero */}
      <div className="about-hero">
        <h1>About <span className="gradient-text">CreditPath AI</span></h1>
        <p>
          An intelligent loan risk prediction system designed to assist both
          borrowers and financial institutions in making smarter, data-driven
          decisions.
        </p>
      </div>

      {/* Cards grid */}
      <div className="about-grid">
        <div className="about-card" style={{ animationDelay: "0s" }}>
          <h3 style={{ fontFamily: "'Inter', sans-serif", color: "var(--text-primary)" }}>What We Do</h3>
          <ul>
            <li>Predict borrower risk using a trained LightGBM AI model</li>
            <li>Provide personalised recommendations for users</li>
            <li>Assist banks in loan approval &amp; recovery decisions</li>
            <li>Visualise risk through interactive charts</li>
          </ul>
        </div>

        <div className="about-card" style={{ animationDelay: "0.08s" }}>
          <h3 style={{ fontFamily: "'Inter', sans-serif", color: "var(--text-primary)" }}>Our Goal</h3>
          <ul>
            <li>Reduce financial risk in the lending ecosystem</li>
            <li>Improve decision-making with data-driven insights</li>
            <li>Make credit analysis accessible to everyone</li>
            <li>Bridge the gap between borrowers and lenders</li>
          </ul>
        </div>

        <div className="about-card" style={{ animationDelay: "0.16s" }}>
          <h3 style={{ fontFamily: "'Inter', sans-serif", color: "var(--text-primary)" }}>Key Features</h3>
          <ul>
            <li>Dual interface for Users &amp; Banks</li>
            <li>Real-time AI predictions (&lt; 1 second)</li>
            <li>Interactive Plotly charts &amp; dashboards</li>
            <li>Personalised financial tips &amp; suggestions</li>
          </ul>
        </div>

        <div className="about-card" style={{ animationDelay: "0.24s" }}>
          <h3 style={{ fontFamily: "'Inter', sans-serif", color: "var(--text-primary)" }}>Future Enhancements</h3>
          <ul>
            <li>Integrate real-time banking API data</li>
            <li>Advanced Explainable AI (XAI) models</li>
            <li>Blockchain-based identity verification</li>
            <li>Mobile Application Deployment</li>
          </ul>
        </div>
      </div>

      {/* Tech Stack */}
      <div className="tech-stack">
        <h2>🛠️ Tech Stack</h2>
        <div className="tech-badges">
          {techStack.map((t, i) => (
            <div className="tech-badge" key={i}>
              <span>{t.icon}</span> {t.label}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default About;